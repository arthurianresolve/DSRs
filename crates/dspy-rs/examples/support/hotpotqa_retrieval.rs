use std::collections::{BTreeMap, BTreeSet};

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
        .filter_map(|(line, (title, _))| selected.contains(title).then_some(*line))
        .collect::<Vec<_>>()
        .join("\n")
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
}
