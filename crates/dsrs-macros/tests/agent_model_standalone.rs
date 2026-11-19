//! Model references on agents bind inside a `#[module]` program.
//! The generated module leaves the standalone function name available.
use dsrs_macros::{agent, tool};

/// Uppercase text.
#[tool]
fn shout(text: String) -> String {
    text.to_uppercase()
}

/// Research the question with the fast model.
#[agent(model = "@fast", tools(shout))]
fn research(question: String) -> String;

// This definition would collide with a standalone function emitted by the macro.
fn research(question: String) -> String {
    question
}

#[test]
fn model_bound_agent_leaves_standalone_function_name_available() {
    let input = research::SigInput {
        question: "hi".to_string(),
    };
    assert_eq!(research(input.question), "hi");
}
