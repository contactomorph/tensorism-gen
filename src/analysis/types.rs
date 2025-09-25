use std::collections::HashMap;

use proc_macro2::Ident;
use syn::Expr;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum HeadKind {
    Tensor,
    Indexer,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct IndexingPosition {
    pub name: Ident,
    pub position: usize,
    pub rank: usize,
    pub kind: HeadKind,
}

impl IndexingPosition {
    pub fn new(name: &Ident, position: usize, rank: usize, kind: HeadKind) -> Self {
        Self {
            name: name.clone(),
            position,
            rank,
            kind,
        }
    }
    pub fn new_indexer_result(name: &Ident) -> Self {
        Self {
            name: name.clone(),
            position: Self::RETURN_POSITION,
            rank: Self::RETURN_POSITION,
            kind: HeadKind::Indexer,
        }
    }
    pub const RETURN_POSITION: usize = usize::MAX;
}

impl PartialEq<(&str, usize, HeadKind)> for IndexingPosition {
    fn eq(&self, other: &(&str, usize, HeadKind)) -> bool {
        self.name == other.0 && self.position == other.1 && self.kind == other.2
    }
}

pub struct IndexingPositionEquivalence {
    pub index: Option<(Ident, String)>,
    pub positions: Vec<IndexingPosition>,
}

impl IndexingPositionEquivalence {
    pub fn new(index: Ident, postfix: String) -> Self {
        Self {
            index: Some((index, postfix)),
            positions: Vec::new(),
        }
    }
    pub fn from_positions(position_a: IndexingPosition, position_b: IndexingPosition) -> Self {
        Self {
            index: None,
            positions: vec![position_a, position_b],
        }
    }
}

pub struct IndexingPositionMapping {
    pub equivalences: Vec<IndexingPositionEquivalence>,
    pub plain_values: HashMap<IndexingPosition, Expr>,
}

impl IndexingPositionMapping {
    pub fn new() -> Self {
        Self {
            equivalences: Vec::new(),
            plain_values: HashMap::new(),
        }
    }
}
