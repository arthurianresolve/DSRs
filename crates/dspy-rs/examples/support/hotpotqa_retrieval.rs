use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Demonstration {
    pub question: String,
    pub context: String,
    pub draft: String,
    pub critique: String,
    pub answer: String,
}

fn terms(text: &str) -> BTreeSet<String> {
    const STOP: &[&str] = &[
        "a", "an", "the", "of", "and", "or", "is", "are", "were", "was", "what", "which", "who",
        "in", "as", "to", "have", "has", "do", "did", "does", "for", "by", "with", "it", "its",
        "from", "that", "this", "on", "at", "be",
    ];
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .filter(|word| !STOP.contains(&word.as_str()))
        .map(|word| word.trim_end_matches('s').to_owned())
        .collect()
}

fn scores(query: &str, documents: &[&str]) -> Vec<f64> {
    let query = terms(query);
    let documents: Vec<_> = documents.iter().map(|text| terms(text)).collect();
    documents
        .iter()
        .map(|document| {
            query
                .intersection(document)
                .map(|word| {
                    let frequency = documents.iter().filter(|doc| doc.contains(word)).count();
                    (1.0 + documents.len() as f64 / (1 + frequency) as f64).ln()
                })
                .sum()
        })
        .collect()
}

/// Select examples using only their questions, never their output labels.
pub fn select_demonstrations<'a>(
    question: &str,
    demos: &'a [Demonstration],
) -> Vec<&'a Demonstration> {
    let questions: Vec<_> = demos.iter().map(|demo| demo.question.as_str()).collect();
    let scores = scores(question, &questions);
    let mut indices: Vec<_> = (0..demos.len()).filter(|&i| scores[i] > 0.0).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| b.cmp(&a)));
    indices.truncate(2);
    indices.sort_unstable();
    indices.into_iter().map(|i| &demos[i]).collect()
}

/// Rank titled context documents and retain ties and documents they reference.
/// Untitled text or a query without shared terms keeps the complete context.
pub fn retrieve_context(question: &str, context: &str) -> String {
    focus_office_evidence(question, &retrieve_documents(question, context))
}

fn retrieve_documents(question: &str, context: &str) -> String {
    let lines: Vec<_> = context.lines().collect();
    let Some(titled): Option<Vec<_>> = lines.iter().map(|line| line.split_once(':')).collect()
    else {
        return context.to_owned();
    };
    if titled.is_empty() {
        return context.to_owned();
    }
    let mut documents: BTreeMap<&str, String> = BTreeMap::new();
    for ((title, _), line) in titled.iter().zip(&lines) {
        let document = documents.entry(title).or_default();
        if !document.is_empty() {
            document.push('\n');
        }
        document.push_str(line);
    }
    let texts: Vec<_> = documents.values().map(String::as_str).collect();
    let ranked = scores(question, &texts);
    let best = ranked.iter().copied().fold(0.0_f64, f64::max);
    if best == 0.0 {
        return context.to_owned();
    }
    let mut selected: BTreeSet<_> = documents
        .keys()
        .copied()
        .zip(ranked)
        .filter_map(|(title, score)| ((score - best).abs() < 1e-9).then_some(title))
        .collect();
    let query = terms(question);
    selected.extend(documents.keys().copied().filter(|title| {
        let title_terms = terms(title);
        !title_terms.is_empty() && title_terms.is_subset(&query)
    }));
    loop {
        let linked: BTreeSet<_> = documents
            .keys()
            .copied()
            .filter(|title| {
                selected.iter().any(|source| {
                    documents[source]
                        .to_lowercase()
                        .contains(&title.to_lowercase())
                })
            })
            .collect();
        if linked.is_subset(&selected) {
            break;
        }
        selected.extend(linked);
    }
    lines
        .iter()
        .zip(titled)
        .filter_map(|(line, (title, _))| {
            let incoming_link = selected
                .iter()
                .any(|selected| line.to_lowercase().contains(&selected.to_lowercase()));
            (selected.contains(title) || incoming_link).then_some(*line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_office_question(question: &str) -> bool {
    static QUESTION: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\b(position|office)\b").unwrap());
    QUESTION.is_match(question)
}

/// For broad position questions, prefer explicit office-holding statements.
/// Appointment questions retain appointment evidence; excluded roles are removed
/// before choosing. Other question types and unrecognized prose keep their context.
fn focus_office_evidence(question: &str, context: &str) -> String {
    static HOLDING: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\b(?:served|serves|serving) as\b").unwrap());
    static APPOINTMENT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\b(?:appointed|named|elected|became)\b").unwrap());
    static SENTENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?]\s+").unwrap());
    static NEGATED: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\b(?:not|never)\b").unwrap());
    static CONJUNCTION: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\band\s+(?:(?:also|later|subsequently|was|were)\s+)*(?:served|serves|serving|appointed|named|elected|became)\b").unwrap()
    });
    if !is_office_question(question) {
        return context.to_owned();
    }
    let mut background = Vec::new();
    let mut facts = Vec::new();
    for line in context.lines() {
        let (title, text) = line
            .split_once(':')
            .map_or((String::new(), line), |(title, text)| {
                (format!("{title}: "), text)
            });
        let mut sentence_starts = vec![0];
        sentence_starts.extend(SENTENCE.find_iter(text).map(|m| m.start() + 1));
        sentence_starts.push(text.len());
        for range in sentence_starts.windows(2) {
            let sentence = text[range[0]..range[1]].trim();
            let mut starts = vec![0];
            starts.extend(CONJUNCTION.find_iter(sentence).map(|m| m.start() + 3));
            starts.push(sentence.len());
            for range in starts.windows(2) {
                let clause = sentence[range[0]..range[1]]
                    .trim()
                    .trim_end_matches(" and")
                    .trim_end();
                let predicate = HOLDING
                    .find(clause)
                    .map(|m| (m, 2))
                    .or_else(|| APPOINTMENT.find(clause).map(|m| (m, 1)));
                let kind = predicate.map_or(0, |(m, kind)| {
                    if NEGATED.is_match(&clause[..m.start()]) {
                        0
                    } else {
                        kind
                    }
                });
                if kind == 0 {
                    background.push(format!("{title}{clause}"));
                } else if validate_exclusion(question, clause).is_ok() {
                    facts.push((kind, format!("{title}{clause}")));
                }
            }
        }
    }
    let Some(mut preferred) = facts.iter().map(|(kind, _)| *kind).max() else {
        return context.to_owned();
    };
    if APPOINTMENT.is_match(question) && facts.iter().any(|(kind, _)| *kind == 1) {
        preferred = 1;
    }
    background.extend(
        facts
            .into_iter()
            .filter_map(|(kind, text)| (kind == preferred).then_some(text)),
    );
    background.join("\n")
}

/// Canonicalize this recipe's office-title output without consulting a reference.
/// Only a trailing recognized territory qualifier is removed; the returned title
/// is always a substring of the model's answer. Unknown qualifiers stay intact.
pub fn canonical_answer<'a>(question: &str, answer: &'a str) -> &'a str {
    static JURISDICTION: LazyLock<Regex> = LazyLock::new(|| {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/cross-framework/hotpotqa/data/territories.json"
        ))
        .unwrap();
        let names = data["main"]["en"]["localeDisplayNames"]["territories"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| {
                key.as_str() != "ZZ"
                    && key.as_bytes().get(..2).is_some_and(|code| {
                        code.iter().all(u8::is_ascii_uppercase)
                            && (key.len() == 2 || key[2..].starts_with("-alt-"))
                    })
            })
            .filter_map(|(_, value)| value.as_str())
            .filter(|name| name.chars().count() > 3)
            .map(regex::escape)
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&format!(
            r"(?i)\s+(?:of|for)\s+(?:the\s+)?(?:{names})\s*\.?$"
        ))
        .unwrap()
    });
    if is_office_question(question)
        && let Some(qualifier) = JURISDICTION.find(answer)
        && !answer[..qualifier.start()].trim().is_empty()
    {
        return answer[..qualifier.start()].trim_end();
    }
    answer
}

/// Enforce a comma-delimited `other than X,` constraint from the question itself.
pub fn validate_exclusion(question: &str, answer: &str) -> Result<(), String> {
    let question = question.to_lowercase();
    let Some((_, tail)) = question.split_once("other than ") else {
        return Ok(());
    };
    let Some((excluded, _)) = tail.split_once(',') else {
        return Ok(());
    };
    let words = |text: &str| {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };
    let excluded_words = words(excluded);
    if !excluded_words.is_empty()
        && words(answer)
            .windows(excluded_words.len())
            .any(|part| part == excluded_words)
    {
        return Err(format!(
            "The question explicitly excludes `{}`. Choose a different answer supported by the context that satisfies this constraint.",
            excluded.trim()
        ));
    }
    Ok(())
}

/// This recipe answers yes/no questions or extracts a phrase from its retrieved context.
pub fn validate_answer(question: &str, context: &str, answer: &str) -> Result<(), String> {
    validate_exclusion(question, answer)?;
    let words = |text: &str| {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };
    let question_words = words(question);
    let answer_words = words(answer);
    let boolean = question_words.first().is_some_and(|word| {
        [
            "is", "are", "was", "were", "do", "does", "did", "can", "has", "have",
        ]
        .contains(&word.as_str())
    });
    if boolean {
        return if answer_words.len() == 1 && ["yes", "no"].contains(&answer_words[0].as_str()) {
            Ok(())
        } else {
            Err("For this yes/no question, return only yes or no.".into())
        };
    }
    if answer_words.is_empty()
        || !words(context)
            .windows(answer_words.len())
            .any(|part| part == answer_words)
    {
        return Err("For this extractive question, answer with a contiguous phrase copied from the context. Do not add an explanation.".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_document_links_for_two_hop_questions() {
        let context = "The Crossing: Lena Marsh played Captain Vale.\nLena Marsh: She served as governor.\nUnrelated: Another person was a singer.";
        assert_eq!(
            retrieve_context(
                "What office did the actor playing Captain Vale hold?",
                context
            ),
            "The Crossing: Lena Marsh played Captain Vale.\nLena Marsh: She served as governor."
        );
    }

    #[test]
    fn keeps_tied_subjects_and_falls_back_without_evidence() {
        let context = "Nira: Nira is Japanese.\nOswin: Oswin is Japanese.\nOther: Other is French.";
        assert_eq!(
            retrieve_context("Compare Nira and Oswin.", context),
            "Nira: Nira is Japanese.\nOswin: Oswin is Japanese."
        );
        assert_eq!(retrieve_context("Unknown subject?", context), context);
        assert_eq!(
            retrieve_context("Nira?", "Nira is Japanese."),
            "Nira is Japanese."
        );
    }

    #[test]
    fn keeps_incoming_series_evidence_without_unrelated_book_paragraphs() {
        let context = "The Glass Ledger: A companion book to Copper Sky.\nThe Glass Ledger: Its hero is a sailor.\nCopper Sky: A fantasy young adult series told in first person.";
        assert_eq!(
            retrieve_context(
                "Which fantasy young adult series is told in first person?",
                context
            ),
            "The Glass Ledger: A companion book to Copper Sky.\nCopper Sky: A fantasy young adult series told in first person."
        );
        assert_eq!(
            retrieve_context(
                "Which fantasy young adult series has The Glass Ledger as a companion book?",
                context
            ),
            context
        );
    }

    #[test]
    fn rejects_explicit_exclusions_without_using_reference_answers() {
        let question = "What office, other than city council member, did she hold?";
        assert!(validate_exclusion(question, "City Council Member").is_err());
        assert!(validate_exclusion(question, "mayor").is_ok());
        assert!(
            validate_exclusion(
                "What role, other than ambassador, did he hold?",
                "ambassadorial adviser"
            )
            .is_ok()
        );
        assert!(validate_exclusion("What role did he hold?", "ambassador").is_ok());
    }

    #[test]
    fn answers_must_fit_the_query_and_retrieved_context() {
        let context = "Copper Sky is a fantasy series.";
        assert!(validate_answer("Which series?", context, "Copper Sky").is_ok());
        assert!(validate_answer("Which series?", context, "The Glass Ledger").is_err());
        assert!(validate_answer("Which series?", context, "no").is_err());
        assert!(validate_answer("Which series?", context, "").is_err());
        assert!(
            validate_answer("Are both Japanese?", "Nira and Oswin are Japanese.", "yes").is_ok()
        );
        assert!(
            validate_answer(
                "Are both Japanese?",
                "Nira and Oswin are Japanese.",
                "Both are Japanese."
            )
            .is_err()
        );
    }

    #[test]
    fn focuses_office_holding_without_losing_the_actor_link() {
        let context = "The Crossing: Lena Marsh played Captain Vale.\nLena Marsh: She was named envoy to Torvia and also served as Treasurer.";
        let focused = retrieve_context(
            "What government position did the woman who played Captain Vale hold?",
            context,
        );
        assert_eq!(
            focused,
            "The Crossing: Lena Marsh played Captain Vale.\nLena Marsh: also served as Treasurer."
        );
        assert!(!focused.contains("envoy"));
    }

    #[test]
    fn keeps_appointments_and_respects_office_exclusions() {
        let context = "She served as governor and was later appointed Education Secretary.";
        assert_eq!(
            focus_office_evidence("What office was she appointed to?", context),
            "was later appointed Education Secretary."
        );
        assert_eq!(
            focus_office_evidence("What office, other than governor, did she hold?", context),
            "was later appointed Education Secretary."
        );
        assert_eq!(
            focus_office_evidence("What position did she hold?", "She was appointed governor."),
            "She was appointed governor."
        );
    }

    #[test]
    fn leaves_acting_roles_and_unknown_office_relations_unchanged() {
        let context =
            "Night Ferry: Dana Keene played Captain Orla.\nDana Keene: She later served as mayor.";
        assert_eq!(
            focus_office_evidence("What role did Dana Keene play?", context),
            context
        );
        assert_eq!(
            focus_office_evidence(
                "What office did she hold?",
                "Her work concerned public health."
            ),
            "Her work concerned public health."
        );
    }

    #[test]
    fn does_not_prefer_negated_office_holding_over_an_appointment() {
        let context = "She never served as governor and was later appointed Treasurer.";
        assert_eq!(
            focus_office_evidence("What office did she hold?", context),
            "She never served as governor\nwas later appointed Treasurer."
        );
    }

    #[test]
    fn canonicalizes_known_jurisdictions_without_removing_title_words() {
        let question = "What office did she hold?";
        assert_eq!(
            canonical_answer(question, "Consul General of Canada"),
            "Consul General"
        );
        assert_eq!(
            canonical_answer(question, "Minister of Defence of the United Kingdom"),
            "Minister of Defence"
        );
        assert_eq!(
            canonical_answer(question, "Consul of Côte d’Ivoire"),
            "Consul"
        );
        for title in [
            "Secretary of the Treasury",
            "Clerk of the House of Lords",
            "Minister of Transport of Eldoria",
            "of Canada",
        ] {
            assert_eq!(canonical_answer(question, title), title);
        }
        assert_eq!(
            canonical_answer("What is the book's title?", "The President of France"),
            "The President of France"
        );
    }
}
