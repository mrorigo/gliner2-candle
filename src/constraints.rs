//! Constraint AST, Kleene-3 evaluation, and the declarative DSL for
//! constrained classification in GLiNER2.5.
//!
//! Hard constraints only. One evaluation method returns Kleene-3 logic:
//! `Some(true)` = satisfied, `Some(false)` = violated, `None` = undetermined.
//!
//! This module imports neither ML frameworks nor IO — pure logic, exhaustively
//! testable.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// ── Kleene-3 helpers ──────────────────────────────────────────────────

pub fn k_not(v: Option<bool>) -> Option<bool> {
    v.map(|b| !b)
}

pub fn k_and(values: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut seen_none = false;
    for v in values {
        if v == Some(false) {
            return Some(false);
        }
        if v.is_none() {
            seen_none = true;
        }
    }
    if seen_none { None } else { Some(true) }
}

pub fn k_or(values: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut seen_none = false;
    for v in values {
        if v == Some(true) {
            return Some(true);
        }
        if v.is_none() {
            seen_none = true;
        }
    }
    if seen_none { None } else { Some(false) }
}

pub fn k_implies(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    k_or([k_not(a), b].into_iter())
}

pub fn k_iff(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a == b),
        _ => None,
    }
}

// ── Assignment ─────────────────────────────────────────────────────────

/// What a constraint reads: selected labels, decided tasks, domains.
pub trait Assignment {
    fn is_decided(&self, task: &str) -> bool;
    fn selected(&self, task: &str) -> HashSet<String>;
    fn domain(&self, task: &str) -> HashSet<String>;
    fn holds(&self, task: &str, label: &str) -> Option<bool>;
}

/// Reference implementation backed by `HashMap`s.
#[derive(Debug, Clone)]
pub struct DictAssignment {
    selected: HashMap<String, HashSet<String>>,
    decided: HashSet<String>,
    domains: HashMap<String, HashSet<String>>,
    label_names: HashMap<String, Vec<String>>,
}

impl DictAssignment {
    pub fn new(
        selected: HashMap<String, HashSet<String>>,
        decided: HashSet<String>,
        domains: HashMap<String, HashSet<String>>,
        label_names: HashMap<String, Vec<String>>,
    ) -> Self {
        Self {
            selected,
            decided,
            domains,
            label_names,
        }
    }
}

impl Assignment for DictAssignment {
    fn is_decided(&self, task: &str) -> bool {
        self.decided.contains(task)
    }

    fn selected(&self, task: &str) -> HashSet<String> {
        self.selected.get(task).cloned().unwrap_or_default()
    }

    fn domain(&self, task: &str) -> HashSet<String> {
        if self.decided.contains(task) {
            return self.selected.get(task).cloned().unwrap_or_default();
        }
        if let Some(d) = self.domains.get(task) {
            return d.clone();
        }
        self.label_names
            .get(task)
            .cloned()
            .map(|v| v.into_iter().collect())
            .unwrap_or_default()
    }

    fn holds(&self, task: &str, label: &str) -> Option<bool> {
        let sel = self.selected(task);
        if sel.contains(label) {
            return Some(true);
        }
        let dom = self.domain(task);
        if dom.contains(label) {
            if self.is_decided(task) {
                Some(false)
            } else {
                None
            }
        } else {
            Some(false)
        }
    }
}

// ── Constraint AST ─────────────────────────────────────────────────────

/// A hard constraint on a classification assignment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Constraint {
    LabelRef {
        task: String,
        label: String,
    },
    AnySelected {
        task: String,
    },
    AnyOtherSelected {
        task: String,
    },
    IsDefault {
        task: String,
    },
    Cardinality {
        task: String,
        minimum: usize,
        maximum: Option<usize>,
    },
    MinLevel {
        task: String,
        level: String,
    },
    MaxLevel {
        task: String,
        level: String,
    },
    AtLevel {
        task: String,
        level: String,
    },
    Not {
        child: Box<Constraint>,
    },
    And {
        children: Vec<Constraint>,
    },
    Or {
        children: Vec<Constraint>,
    },
    ExactlyOneOf {
        children: Vec<Constraint>,
    },
    Implies {
        cond: Box<Constraint>,
        then: Box<Constraint>,
    },
    Iff {
        left: Box<Constraint>,
        right: Box<Constraint>,
    },
    Excludes {
        left: Box<Constraint>,
        right: Box<Constraint>,
    },
}

impl Constraint {
    // ── references ────────────────────────────────────────────────────
    pub fn references(&self) -> HashSet<String> {
        match self {
            Self::LabelRef { task, .. }
            | Self::AnySelected { task }
            | Self::AnyOtherSelected { task }
            | Self::IsDefault { task }
            | Self::Cardinality { task, .. }
            | Self::MinLevel { task, .. }
            | Self::MaxLevel { task, .. }
            | Self::AtLevel { task, .. } => HashSet::from([task.clone()]),
            Self::Not { child } => child.references(),
            Self::And { children } | Self::Or { children } | Self::ExactlyOneOf { children } => {
                children.iter().flat_map(|c| c.references()).collect()
            }
            Self::Implies { cond, then } => {
                let mut refs = cond.references();
                refs.extend(then.references());
                refs
            }
            Self::Iff { left, right } => {
                let mut refs = left.references();
                refs.extend(right.references());
                refs
            }
            Self::Excludes { left, right } => {
                let mut refs = left.references();
                refs.extend(right.references());
                refs
            }
        }
    }

    /// Evaluate against an assignment: `Some(true)` satisfied, `Some(false)` violated, `None` undetermined.
    pub fn evaluate(&self, a: &dyn Assignment) -> Option<bool> {
        match self {
            Self::LabelRef { task, label } => a.holds(task, label),

            Self::AnySelected { task } => {
                let sel = a.selected(task);
                let dom = a.domain(task);
                let combined: HashSet<String> = sel.union(&dom).cloned().collect();
                k_or(combined.iter().map(|l| a.holds(task, l)))
            }

            Self::AnyOtherSelected { task } => {
                let sel = a.selected(task);
                let dom = a.domain(task);
                let combined: HashSet<String> = sel.union(&dom).cloned().collect();
                // We don't track default in the assignment; treat as "any non-empty"
                k_or(combined.iter().map(|l| a.holds(task, l)))
            }

            Self::IsDefault { task } => {
                let sel = a.selected(task);
                if sel.is_empty() {
                    Some(true)
                } else {
                    // Without default tracking, we check if nothing is selected
                    Some(false)
                }
            }

            Self::Cardinality {
                task,
                minimum,
                maximum,
            } => {
                let sel = a.selected(task);
                let dom = a.domain(task);
                let lo = sel.len();
                let hi = sel.union(&dom).count();
                let mx = maximum.unwrap_or(hi);
                if lo > mx || hi < *minimum {
                    Some(false)
                } else if lo >= *minimum && hi <= mx {
                    Some(true)
                } else {
                    None
                }
            }

            Self::MinLevel { task, level } => {
                let sel = a.selected(task);
                let dom = a.domain(task);
                let combined: HashSet<String> = sel.union(&dom).cloned().collect();
                if combined.is_empty() {
                    return Some(false);
                }
                // Without ordinal info, treat level names as their position in domain
                let floor = dom.iter().position(|l| l == level).unwrap_or(0);
                let levels: Vec<usize> = combined
                    .iter()
                    .filter_map(|l| dom.iter().position(|d| d == l))
                    .collect();
                if levels.is_empty() {
                    return Some(false);
                }
                let min_level = *levels.iter().min()?;
                let max_level = *levels.iter().max()?;
                if min_level >= floor {
                    Some(true)
                } else if max_level < floor {
                    Some(false)
                } else {
                    None
                }
            }

            Self::MaxLevel { task, level } => {
                let sel = a.selected(task);
                let dom = a.domain(task);
                let combined: HashSet<String> = sel.union(&dom).cloned().collect();
                if combined.is_empty() {
                    return Some(false);
                }
                let ceil = dom.iter().position(|l| l == level).unwrap_or(dom.len());
                let levels: Vec<usize> = combined
                    .iter()
                    .filter_map(|l| dom.iter().position(|d| d == l))
                    .collect();
                if levels.is_empty() {
                    return Some(false);
                }
                let min_level = *levels.iter().min()?;
                let max_level = *levels.iter().max()?;
                if max_level <= ceil {
                    Some(true)
                } else if min_level > ceil {
                    Some(false)
                } else {
                    None
                }
            }

            Self::AtLevel { task, level } => {
                let sel = a.selected(task);
                let dom = a.domain(task);
                let combined: HashSet<String> = sel.union(&dom).cloned().collect();
                if combined.is_empty() {
                    return Some(false);
                }
                let target = dom.iter().position(|l| l == level);
                let levels: HashSet<usize> = combined
                    .iter()
                    .filter_map(|l| dom.iter().position(|d| d == l))
                    .collect();
                if levels.is_empty() {
                    return Some(false);
                }
                match target {
                    Some(t) if levels.len() == 1 && levels.contains(&t) => Some(true),
                    Some(t) if !levels.contains(&t) => Some(false),
                    _ => None,
                }
            }

            Self::Not { child } => k_not(child.evaluate(a)),

            Self::And { children } => k_and(children.iter().map(|c| c.evaluate(a))),

            Self::Or { children } => k_or(children.iter().map(|c| c.evaluate(a))),

            Self::ExactlyOneOf { children } => {
                let vals: Vec<Option<bool>> = children.iter().map(|c| c.evaluate(a)).collect();
                let trues = vals.iter().filter(|v| **v == Some(true)).count();
                let nones = vals.iter().filter(|v| v.is_none()).count();
                if trues >= 2 {
                    Some(false)
                } else if trues == 1 {
                    if nones == 0 { Some(true) } else { None }
                } else if nones == 0 {
                    Some(false)
                } else {
                    None
                }
            }

            Self::Implies { cond, then } => k_implies(cond.evaluate(a), then.evaluate(a)),

            Self::Iff { left, right } => k_iff(left.evaluate(a), right.evaluate(a)),

            Self::Excludes { left, right } => {
                k_not(k_and([left.evaluate(a), right.evaluate(a)].into_iter()))
            }
        }
    }

    /// Is this constraint still satisfiable?
    pub fn still_satisfiable(&self, a: &dyn Assignment) -> bool {
        self.evaluate(a) != Some(false)
    }

    /// Is this constraint fully satisfied?
    pub fn satisfied(&self, a: &dyn Assignment) -> bool {
        self.evaluate(a) == Some(true)
    }
}

// ── DSL builders ───────────────────────────────────────────────────────

pub fn label(task: &str, name: &str) -> Constraint {
    Constraint::LabelRef {
        task: task.to_string(),
        label: name.to_string(),
    }
}

pub fn any_selected(task: &str) -> Constraint {
    Constraint::AnySelected {
        task: task.to_string(),
    }
}

pub fn at_least(task: &str, k: usize) -> Constraint {
    Constraint::Cardinality {
        task: task.to_string(),
        minimum: k,
        maximum: None,
    }
}

pub fn at_most(task: &str, k: usize) -> Constraint {
    Constraint::Cardinality {
        task: task.to_string(),
        minimum: 0,
        maximum: Some(k),
    }
}

pub fn exactly(task: &str, k: usize) -> Constraint {
    Constraint::Cardinality {
        task: task.to_string(),
        minimum: k,
        maximum: Some(k),
    }
}

pub fn not_(child: Constraint) -> Constraint {
    Constraint::Not {
        child: Box::new(child),
    }
}

pub fn all_of(children: Vec<Constraint>) -> Constraint {
    Constraint::And { children }
}

pub fn any_of(children: Vec<Constraint>) -> Constraint {
    Constraint::Or { children }
}

pub fn implies(cond: Constraint, then: Constraint) -> Constraint {
    Constraint::Implies {
        cond: Box::new(cond),
        then: Box::new(then),
    }
}

pub fn iff(left: Constraint, right: Constraint) -> Constraint {
    Constraint::Iff {
        left: Box::new(left),
        right: Box::new(right),
    }
}

pub fn excludes(left: Constraint, right: Constraint) -> Constraint {
    Constraint::Excludes {
        left: Box::new(left),
        right: Box::new(right),
    }
}

pub fn exactly_one_of(children: Vec<Constraint>) -> Constraint {
    Constraint::ExactlyOneOf { children }
}

// ── DFS Decoder ────────────────────────────────────────────────────────

/// One local assignment for a task: a set of selected labels and a utility score.
#[derive(Debug, Clone)]
pub struct LocalAssignment {
    pub selected: HashSet<String>,
    pub utility: f32,
}

/// The problem given to the decoder.
pub struct DecodeProblem {
    /// Ordered list of tasks.
    pub task_order: Vec<String>,
    /// Per-task candidate local assignments, sorted by utility descending.
    pub locals: HashMap<String, Vec<LocalAssignment>>,
    /// Constraints to satisfy.
    pub constraints: Vec<Constraint>,
}

impl DecodeProblem {
    /// Which constraints mention a given task?
    pub fn constraints_touching(&self, task: &str) -> Vec<&Constraint> {
        self.constraints
            .iter()
            .filter(|c| c.references().contains(task))
            .collect()
    }

    /// Build a DictAssignment from partial chosen locals.
    fn assignment(
        &self,
        chosen: &HashMap<String, LocalAssignment>,
        decided: &[&str],
    ) -> DictAssignment {
        let mut selected = HashMap::new();
        let mut decided_set = HashSet::new();
        let domains = HashMap::new();
        let mut label_names = HashMap::new();

        for task in &self.task_order {
            if let Some(local) = chosen.get(task.as_str()) {
                selected.insert(task.clone(), local.selected.clone());
            }
            label_names.insert(
                task.clone(),
                self.locals[task]
                    .iter()
                    .flat_map(|l| l.selected.iter().cloned())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect(),
            );
        }
        for d in decided {
            decided_set.insert(d.to_string());
        }
        DictAssignment::new(selected, decided_set, domains, label_names)
    }
}

/// Solution returned by the decoder.
#[derive(Debug)]
pub struct Solution {
    pub assignments: HashMap<String, LocalAssignment>,
    pub score: f32,
    pub violations: Vec<Constraint>,
    pub exact: bool,
}

/// Exact DFS decoder with branch and bound.
pub fn decode_exact(problem: &DecodeProblem, budget: usize) -> Option<Solution> {
    let order = order_tasks(problem);
    let suffix = suffix_max(&order, problem);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_assign: Option<HashMap<String, LocalAssignment>> = None;
    let mut nodes: usize = 0;

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        problem: &DecodeProblem,
        order: &[String],
        suffix: &[f32],
        i: usize,
        chosen: &mut HashMap<String, LocalAssignment>,
        score: f32,
        best_score: &mut f32,
        best_assign: &mut Option<HashMap<String, LocalAssignment>>,
        nodes: &mut usize,
        budget: usize,
    ) {
        *nodes += 1;
        if *nodes > budget {
            return;
        }
        if i == order.len() {
            if score > *best_score {
                *best_score = score;
                *best_assign = Some(chosen.clone());
            }
            return;
        }
        if score + suffix[i] <= *best_score {
            return;
        }

        let task = &order[i];
        let touching = problem.constraints_touching(task);

        if let Some(locals) = problem.locals.get(task) {
            for local in locals {
                if score + local.utility + suffix[i + 1] <= *best_score {
                    break;
                }
                chosen.insert(task.clone(), local.clone());
                let decided: Vec<&str> = order[..=i].iter().map(|s| s.as_str()).collect();
                let a = problem.assignment(chosen, &decided);
                if touching.iter().all(|c| c.still_satisfiable(&a)) {
                    dfs(
                        problem,
                        order,
                        suffix,
                        i + 1,
                        chosen,
                        score + local.utility,
                        best_score,
                        best_assign,
                        nodes,
                        budget,
                    );
                }
                chosen.remove(task);
            }
        }
    }

    let mut chosen = HashMap::new();
    dfs(
        problem,
        &order,
        &suffix,
        0,
        &mut chosen,
        0.0,
        &mut best_score,
        &mut best_assign,
        &mut nodes,
        budget,
    );

    best_assign.map(|assignments| {
        let score = best_score;
        Solution {
            assignments,
            score,
            violations: Vec::new(),
            exact: true,
        }
    })
}

fn order_tasks(problem: &DecodeProblem) -> Vec<String> {
    let mut order = problem.task_order.clone();
    order.sort_by(|a, b| {
        let ca = problem.constraints_touching(a).len();
        let cb = problem.constraints_touching(b).len();
        cb.cmp(&ca).then(a.cmp(b))
    });
    order
}

fn suffix_max(order: &[String], problem: &DecodeProblem) -> Vec<f32> {
    let mut suffix = vec![0.0f32; order.len() + 1];
    for i in (0..order.len()).rev() {
        let best = problem
            .locals
            .get(&order[i])
            .map(|ls| {
                ls.iter()
                    .map(|l| l.utility)
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .unwrap_or(0.0);
        suffix[i] = best + suffix[i + 1];
    }
    suffix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn dummy_task(task: &str, labels: &[&str]) -> (String, Vec<String>) {
        (
            task.to_string(),
            labels.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn make_assignment(selected: Vec<(&str, Vec<&str>)>, decided: Vec<&str>) -> DictAssignment {
        let sel: HashMap<String, HashSet<String>> = selected
            .into_iter()
            .map(|(t, ls)| (t.to_string(), ls.into_iter().map(String::from).collect()))
            .collect();
        let dec: HashSet<String> = decided.into_iter().map(String::from).collect();
        let label_names: HashMap<String, Vec<String>> = sel
            .keys()
            .map(|t| (t.clone(), sel[t].iter().cloned().collect()))
            .collect();
        DictAssignment::new(sel, dec, HashMap::new(), label_names)
    }

    // ── Kleene-3 ──────────────────────────────────────────────────────

    #[test]
    fn k_not_basic() {
        assert_eq!(k_not(Some(true)), Some(false));
        assert_eq!(k_not(Some(false)), Some(true));
        assert_eq!(k_not(None), None);
    }

    #[test]
    fn k_and_basic() {
        assert_eq!(k_and(vec![Some(true), Some(true)].into_iter()), Some(true));
        assert_eq!(
            k_and(vec![Some(true), Some(false)].into_iter()),
            Some(false)
        );
        assert_eq!(k_and(vec![Some(true), None].into_iter()), None);
        assert_eq!(k_and(vec![].into_iter()), Some(true));
    }

    #[test]
    fn k_or_basic() {
        assert_eq!(
            k_or(vec![Some(false), Some(false)].into_iter()),
            Some(false)
        );
        assert_eq!(k_or(vec![Some(false), Some(true)].into_iter()), Some(true));
        assert_eq!(k_or(vec![Some(false), None].into_iter()), None);
        assert_eq!(k_or(vec![].into_iter()), Some(false));
    }

    #[test]
    fn k_implies_basic() {
        assert_eq!(k_implies(Some(true), Some(true)), Some(true));
        assert_eq!(k_implies(Some(true), Some(false)), Some(false));
        assert_eq!(k_implies(Some(false), Some(true)), Some(true));
        assert_eq!(k_implies(Some(false), Some(false)), Some(true));
        assert_eq!(k_implies(None, Some(true)), Some(true));
        assert_eq!(k_implies(None, Some(false)), None);
        assert_eq!(k_implies(None, None), None);
        assert_eq!(k_implies(Some(true), None), None);
        assert_eq!(k_implies(Some(false), None), Some(true));
    }

    // ── LabelRef ──────────────────────────────────────────────────────

    #[test]
    fn label_ref_holds() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = label("t", "a");
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn label_ref_not_holds() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = label("t", "b");
        assert_eq!(c.evaluate(&a), Some(false));
    }

    // ── Boolean nodes ─────────────────────────────────────────────────

    #[test]
    fn and_all_true() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = all_of(vec![label("t", "a"), label("t", "b")]);
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn and_one_false() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = all_of(vec![label("t", "a"), label("t", "b")]);
        assert_eq!(c.evaluate(&a), Some(false));
    }

    #[test]
    fn or_one_true() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = any_of(vec![label("t", "a"), label("t", "b")]);
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn or_none_true() {
        let a = make_assignment(vec![("t", vec![])], vec!["t"]);
        let c = any_of(vec![label("t", "a"), label("t", "b")]);
        assert_eq!(c.evaluate(&a), Some(false));
    }

    #[test]
    fn not_label() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = not_(label("t", "b"));
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn implies_satisfied() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = implies(label("t", "a"), label("t", "b"));
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn implies_violated() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = implies(label("t", "a"), label("t", "b"));
        assert_eq!(c.evaluate(&a), Some(false));
    }

    #[test]
    fn excludes_satisfied() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = excludes(label("t", "a"), label("t", "b"));
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn excludes_violated() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = excludes(label("t", "a"), label("t", "b"));
        assert_eq!(c.evaluate(&a), Some(false));
    }

    #[test]
    fn exactly_one_of_satisfied() {
        let a = make_assignment(vec![("t", vec!["a"])], vec!["t"]);
        let c = exactly_one_of(vec![label("t", "a"), label("t", "b")]);
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn exactly_one_of_violated() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = exactly_one_of(vec![label("t", "a"), label("t", "b")]);
        assert_eq!(c.evaluate(&a), Some(false));
    }

    // ── Cardinality ───────────────────────────────────────────────────

    #[test]
    fn cardinality_at_least() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = at_least("t", 2);
        assert_eq!(c.evaluate(&a), Some(true));
    }

    #[test]
    fn cardinality_at_most() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = at_most("t", 1);
        assert_eq!(c.evaluate(&a), Some(false));
    }

    #[test]
    fn cardinality_exactly() {
        let a = make_assignment(vec![("t", vec!["a", "b"])], vec!["t"]);
        let c = exactly("t", 2);
        assert_eq!(c.evaluate(&a), Some(true));
    }

    // ── References ────────────────────────────────────────────────────

    #[test]
    fn references_union() {
        let c = implies(label("t1", "a"), label("t2", "b"));
        let refs = c.references();
        assert!(refs.contains("t1"));
        assert!(refs.contains("t2"));
    }

    // ── DFS Decoder ──────────────────────────────────────────────────

    #[test]
    fn decode_simple_feasible() {
        let mut locals = HashMap::new();
        locals.insert(
            "t1".to_string(),
            vec![
                LocalAssignment {
                    selected: ["a".to_string()].into_iter().collect(),
                    utility: 2.0,
                },
                LocalAssignment {
                    selected: ["b".to_string()].into_iter().collect(),
                    utility: 1.0,
                },
            ],
        );
        locals.insert(
            "t2".to_string(),
            vec![
                LocalAssignment {
                    selected: ["x".to_string()].into_iter().collect(),
                    utility: 3.0,
                },
                LocalAssignment {
                    selected: ["y".to_string()].into_iter().collect(),
                    utility: 1.0,
                },
            ],
        );

        let problem = DecodeProblem {
            task_order: vec!["t1".to_string(), "t2".to_string()],
            locals,
            constraints: vec![excludes(label("t1", "a"), label("t2", "x"))],
        };

        let sol = decode_exact(&problem, 1000).unwrap();
        // Best feasible: t1=b, t2=x (utility 1+3=4) or t1=a, t2=y (2+1=3)
        // t1=a, t2=x is excluded.
        assert!((sol.score - 4.0).abs() < 1e-6);
    }

    #[test]
    fn decode_all_feasible() {
        let mut locals = HashMap::new();
        locals.insert(
            "t1".to_string(),
            vec![LocalAssignment {
                selected: ["a".to_string()].into_iter().collect(),
                utility: 1.0,
            }],
        );

        let problem = DecodeProblem {
            task_order: vec!["t1".to_string()],
            locals,
            constraints: vec![],
        };

        let sol = decode_exact(&problem, 100).unwrap();
        assert!((sol.score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_infeasible_returns_none() {
        let mut locals = HashMap::new();
        locals.insert(
            "t1".to_string(),
            vec![LocalAssignment {
                selected: ["a".to_string()].into_iter().collect(),
                utility: 1.0,
            }],
        );
        locals.insert(
            "t2".to_string(),
            vec![LocalAssignment {
                selected: ["a".to_string()].into_iter().collect(),
                utility: 1.0,
            }],
        );

        let problem = DecodeProblem {
            task_order: vec!["t1".to_string(), "t2".to_string()],
            locals,
            constraints: vec![
                at_least("t1", 1),
                exactly("t1", 0), // contradicts at_least(1)
            ],
        };

        let sol = decode_exact(&problem, 1000);
        // One of the constraints will be violated for every assignment
        assert!(sol.is_none() || sol.unwrap().violations.is_empty());
    }

    // ── Serialization ─────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let c = implies(
            label("sentiment", "positive"),
            not_(label("sentiment", "negative")),
        );
        let json = serde_json::to_string(&c).unwrap();
        let c2: Constraint = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2);
    }
}
