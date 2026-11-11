use std::collections::HashSet;

use anyhow::{Result, ensure};
use serde::Deserialize;

pub const SUPPORTED_METRIC: &str = "context_supported_normalized_exact_match";

pub fn normalized(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .filter(|word| !["a", "an", "the"].contains(word))
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Deserialize)]
pub struct SupportedAnswerAnnotations {
    schema_version: u32,
    annotation_set_id: String,
    metric: String,
    description: String,
    examples: Vec<SupportedAnswerAnnotation>,
}

#[derive(Debug, Deserialize)]
struct SupportedAnswerAnnotation {
    id: String,
    question: String,
    context: String,
    reference_answer: String,
    accepted_answers: Vec<String>,
    rationale: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AnswerScore {
    pub exact_match: bool,
    pub supported_answer_match: bool,
    pub annotation_applied: bool,
    pub accepted_answers: Vec<String>,
}

impl SupportedAnswerAnnotations {
    pub fn from_json(text: &str) -> Result<Self> {
        let annotations: Self = serde_json::from_str(text)?;
        ensure!(
            annotations.schema_version == 1,
            "unsupported answer-annotation schema version {}",
            annotations.schema_version
        );
        ensure!(
            !annotations.annotation_set_id.trim().is_empty(),
            "answer-annotation set id must not be empty"
        );
        ensure!(
            annotations.metric == SUPPORTED_METRIC,
            "unsupported answer-annotation metric `{}`",
            annotations.metric
        );
        ensure!(
            !annotations.description.trim().is_empty(),
            "answer-annotation description must not be empty"
        );

        let mut ids = HashSet::new();
        for annotation in &annotations.examples {
            ensure!(
                ids.insert(annotation.id.as_str()),
                "duplicate answer annotation for `{}`",
                annotation.id
            );
            ensure!(
                !annotation.question.trim().is_empty()
                    && !annotation.context.trim().is_empty()
                    && !annotation.reference_answer.trim().is_empty(),
                "answer annotation `{}` must freeze its question, context and reference",
                annotation.id
            );
            ensure!(
                !annotation.rationale.trim().is_empty(),
                "answer annotation `{}` must explain why alternatives are supported",
                annotation.id
            );
            ensure!(
                !annotation.accepted_answers.is_empty(),
                "answer annotation `{}` has no accepted answers",
                annotation.id
            );
            let mut answers = HashSet::new();
            for answer in &annotation.accepted_answers {
                let answer = normalized(answer);
                ensure!(
                    !answer.is_empty(),
                    "answer annotation `{}` contains an empty normalized answer",
                    annotation.id
                );
                ensure!(
                    answers.insert(answer),
                    "answer annotation `{}` contains duplicate normalized answers",
                    annotation.id
                );
            }
            ensure!(
                answers.contains(&normalized(&annotation.reference_answer)),
                "answer annotation `{}` must retain its strict reference answer",
                annotation.id
            );
        }
        Ok(annotations)
    }

    pub fn id(&self) -> &str {
        &self.annotation_set_id
    }

    pub fn score(
        &self,
        id: &str,
        question: &str,
        context: &str,
        reference_answer: &str,
        answer: &str,
    ) -> Result<AnswerScore> {
        let annotation = self.examples.iter().find(|annotation| annotation.id == id);
        let accepted_answers = if let Some(annotation) = annotation {
            ensure!(
                annotation.question == question
                    && annotation.context == context
                    && annotation.reference_answer == reference_answer,
                "answer annotation `{id}` does not match the frozen question, context and reference"
            );
            annotation.accepted_answers.clone()
        } else {
            vec![reference_answer.to_owned()]
        };
        let answer = normalized(answer);
        Ok(AnswerScore {
            exact_match: answer == normalized(reference_answer),
            supported_answer_match: accepted_answers
                .iter()
                .any(|accepted| answer == normalized(accepted)),
            annotation_applied: annotation.is_some(),
            accepted_answers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANNOTATIONS: &str = r#"{
        "schema_version": 1,
        "annotation_set_id": "test-supported-answers-v1",
        "metric": "context_supported_normalized_exact_match",
        "description": "test annotations",
        "examples": [{
            "id": "office",
            "question": "Which office?",
            "context": "She was named ambassador and served as treasurer.",
            "reference_answer": "treasurer",
            "accepted_answers": ["treasurer", "ambassador"],
            "rationale": "Both offices are stated."
        }]
    }"#;

    #[test]
    fn preserves_strict_score_while_accepting_reviewed_alternative() {
        let annotations = SupportedAnswerAnnotations::from_json(ANNOTATIONS).unwrap();
        let score = annotations
            .score(
                "office",
                "Which office?",
                "She was named ambassador and served as treasurer.",
                "treasurer",
                "Ambassador",
            )
            .unwrap();
        assert_eq!(
            score,
            AnswerScore {
                exact_match: false,
                supported_answer_match: true,
                annotation_applied: true,
                accepted_answers: vec!["treasurer".into(), "ambassador".into()],
            }
        );
    }

    #[test]
    fn unannotated_examples_use_only_the_strict_reference() {
        let annotations = SupportedAnswerAnnotations::from_json(ANNOTATIONS).unwrap();
        let score = annotations
            .score(
                "other",
                "Which series?",
                "Copper Sky is a series.",
                "Copper Sky",
                "Copper Sky",
            )
            .unwrap();
        assert!(score.exact_match && score.supported_answer_match);
        assert!(!score.annotation_applied);
        assert_eq!(score.accepted_answers, ["Copper Sky"]);
    }

    #[test]
    fn rejects_stale_or_malformed_annotations() {
        let annotations = SupportedAnswerAnnotations::from_json(ANNOTATIONS).unwrap();
        assert!(
            annotations
                .score(
                    "office",
                    "Which office changed?",
                    "She was named ambassador and served as treasurer.",
                    "treasurer",
                    "ambassador",
                )
                .is_err()
        );
        let duplicate = ANNOTATIONS.replace(
            "\"treasurer\", \"ambassador\"",
            "\"treasurer\", \"the treasurer\"",
        );
        assert!(SupportedAnswerAnnotations::from_json(&duplicate).is_err());
    }
}
