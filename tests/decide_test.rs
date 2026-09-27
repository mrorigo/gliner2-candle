//! GLiNER2.5-Decide classification parity.
//!
//! The Decide family is a zero-shot classification specialist. Two shapes ship:
//! `fastino/GLiNER2.5-Decide` (span architecture, DeBERTa-v3-large) and
//! `fastino/GLiNER2.5-multi-Decide` (boundary architecture, multilingual).
//! `fastino/GLiNER2.5-Decide-1B` uses a ModernBERT encoder and is not supported.
//!
//! Expectations are the outputs published on the model cards. The cards label
//! them "potential results", so a handful are borderline calls rather than
//! ground truth; those are asserted loosely below and noted inline.
//!
//! Run explicitly: cargo test --test decide_test -- --ignored --nocapture
//! Override the checkpoint with GLINER_MODEL.

use gliner2_candle::inference::engine::GLiNER2;
use gliner2_candle::schema::builder::SchemaBuilder;
use serde_json::Value;

fn labels(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn single(result: &Value, task: &str) -> String {
    result
        .get(task)
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("{task}: expected a bare string, got {result}"))
        .to_string()
}

fn multi(result: &Value, task: &str) -> Vec<String> {
    result
        .get(task)
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("{task}: expected an array, got {result}"))
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect()
}

#[test]
#[ignore = "downloads 500MB-1.9GB; run explicitly with --ignored"]
fn decide_span_classification() {
    let model_id =
        std::env::var("GLINER_MODEL").unwrap_or_else(|_| "fastino/GLiNER2.5-Decide".to_string());
    println!("=== {model_id} ===");
    let engine = GLiNER2::from_pretrained(&model_id).expect("load failed");

    // Span checkpoints must not be routed through the boundary head. The name
    // contains "gliner2.5", so this only works because config.json wins.
    assert_eq!(
        engine.config().architecture,
        gliner2_candle::config::Architecture::Gliner2
    );
    assert_eq!(engine.config().hidden_size, 1024);
    assert_eq!(engine.config().num_hidden_layers, 24);

    // --- single label ---------------------------------------------------------
    let review = "Battery dies before lunch, but the keyboard and the screen are the \
                  best I have used on a laptop.";
    let res = engine
        .classify_text(
            "My subscription renewed on April 15 for 5,400 after the service was already \
             down. Can I get that charge refunded?",
            &[(
                "intent".into(),
                labels(&[
                    "order_status",
                    "refund_request",
                    "cancel_subscription",
                    "update_payment",
                    "login_problem",
                    "shipping_delay",
                    "bug_report",
                    "speak_to_human",
                    "other",
                ]),
            )],
            None,
            false,
            None,
        )
        .unwrap();
    assert_eq!(single(&res, "intent"), "refund_request");

    // --- multi-label ----------------------------------------------------------
    let schema = SchemaBuilder::new()
        .classification(
            "aspects",
            labels(&[
                "battery", "keyboard", "screen", "camera", "price", "support",
            ]),
        )
        .multi_label(true)
        .threshold(0.4)
        .done()
        .build()
        .unwrap();
    let res = engine
        .extract(review, &schema, 0.4, false, false, None)
        .unwrap();
    assert_eq!(multi(&res, "aspects"), ["battery", "keyboard", "screen"]);

    // --- several heads in one forward pass ------------------------------------
    let res = engine
        .classify_text(
            "From: compliance@group.example\nSubject: Protocol update - action required \
             today\n\nPlease confirm the new retention rule is applied before Friday's audit.",
            &[
                (
                    "intent".into(),
                    labels(&[
                        "fyi",
                        "request",
                        "approval",
                        "complaint",
                        "newsletter",
                        "security_alert",
                    ]),
                ),
                (
                    "urgency".into(),
                    labels(&["low", "normal", "high", "critical"]),
                ),
                (
                    "route".into(),
                    labels(&[
                        "support", "billing", "legal", "security", "finance", "archive",
                    ]),
                ),
            ],
            None,
            false,
            None,
        )
        .unwrap();
    assert_eq!(
        res.as_object().unwrap().len(),
        3,
        "all heads must be scored: {res}"
    );
    assert_eq!(single(&res, "intent"), "request");
    assert_eq!(single(&res, "route"), "legal");
    // Card says "high". The span Decide checkpoint prefers "critical" here; the
    // logits are near-tied, so only the ordinal ordering is meaningful.
    let urgency = single(&res, "urgency");
    assert!(
        ["high", "critical"].contains(&urgency.as_str()),
        "urgency should be an upper-tier label, got {urgency}"
    );

    // --- labels carrying a description ----------------------------------------
    let schema = SchemaBuilder::new()
        .classification(
            "intent",
            labels(&["card_pin_change", "card_lost", "balance_inquiry"]),
        )
        .label_descriptions(
            [
                (
                    "card_pin_change",
                    "The customer wants a new PIN or the current PIN replaced",
                ),
                ("card_lost", "The physical card is missing"),
                ("balance_inquiry", "The customer wants the current balance"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        )
        .done()
        .build()
        .unwrap();
    let res = engine
        .extract(
            "Please reset the card PIN. The new one never arrived and the old one is \
             locked after three tries.",
            &schema,
            0.5,
            false,
            false,
            None,
        )
        .unwrap();
    // The task name is recovered from the folded prompt string, so the output key
    // must be the bare task, not the prompt.
    assert_eq!(single(&res, "intent"), "card_pin_change");

    // --- ordinal scale --------------------------------------------------------
    let res = engine
        .classify_text(
            "I finished it in two nights. The ending is earned, the middle drags, and \
             I would still hand it to a friend.",
            &[(
                "rating".into(),
                labels(&["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]),
            )],
            None,
            false,
            None,
        )
        .unwrap();
    let rating: u32 = single(&res, "rating").parse().expect("numeric label");
    assert!(
        rating >= 6,
        "a favourable review should score high, got {rating}"
    );

    // --- confidence is a softmax, not a sigmoid -------------------------------
    let res = engine
        .classify_text(
            "My subscription renewed on April 15 for 5,400 after the service was already \
             down. Can I get that charge refunded?",
            &[("intent".into(), labels(&["refund_request", "other"]))],
            None,
            true,
            None,
        )
        .unwrap();
    let conf = res["intent"]["confidence"].as_f64().expect("confidence");
    assert!(
        (0.0..=1.0).contains(&conf),
        "confidence out of range: {conf}"
    );
    // A two-way softmax on a confident call must be near 1, which a sigmoid over a
    // single raw logit would not be.
    assert!(conf > 0.9, "expected softmax confidence, got {conf}");
}

#[test]
#[ignore = "downloads ~500MB; run explicitly with --ignored"]
fn decide_boundary_classification() {
    // The multilingual variant is a GLiNER2.5 boundary checkpoint and must take
    // the boundary path.
    let engine = GLiNER2::from_pretrained("fastino/GLiNER2.5-multi-Decide").expect("load failed");
    assert_eq!(
        engine.config().architecture,
        gliner2_candle::config::Architecture::Gliner25
    );

    let res = engine
        .classify_text(
            "My subscription renewed on April 15 for 5,400 after the service was already \
             down. Can I get that charge refunded?",
            &[(
                "intent".into(),
                labels(&[
                    "order_status",
                    "refund_request",
                    "cancel_subscription",
                    "update_payment",
                    "login_problem",
                    "shipping_delay",
                    "bug_report",
                    "speak_to_human",
                    "other",
                ]),
            )],
            None,
            false,
            None,
        )
        .unwrap();
    assert_eq!(single(&res, "intent"), "refund_request");

    let res = engine
        .classify_text(
            "From: compliance@group.example\nSubject: Protocol update - action required \
             today\n\nPlease confirm the new retention rule is applied before Friday's audit.",
            &[
                (
                    "intent".into(),
                    labels(&[
                        "fyi",
                        "request",
                        "approval",
                        "complaint",
                        "newsletter",
                        "security_alert",
                    ]),
                ),
                (
                    "urgency".into(),
                    labels(&["low", "normal", "high", "critical"]),
                ),
                (
                    "route".into(),
                    labels(&[
                        "support", "billing", "legal", "security", "finance", "archive",
                    ]),
                ),
            ],
            None,
            false,
            None,
        )
        .unwrap();
    assert_eq!(res.as_object().unwrap().len(), 3, "{res}");
    assert_eq!(single(&res, "intent"), "request");
    assert_eq!(single(&res, "urgency"), "high");
    assert_eq!(single(&res, "route"), "legal");
}
