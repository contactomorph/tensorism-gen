use std::{collections::HashMap, ops::Deref};

use proc_macro2::Ident;
use syn::Expr;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum HeadKind {
    Tensor,
    Indexer,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum IndexingPositionContent {
    TensorSingleIndex,
    TensorSpecificIndex(usize),
    IndexerSpecificIndex(usize),
    IndexerResult,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct IndexingPosition {
    pub name: Ident,
    pub content: IndexingPositionContent,
}

#[cfg(test)]
impl PartialEq<(&str, usize, HeadKind)> for IndexingPosition {
    fn eq(&self, other: &(&str, usize, HeadKind)) -> bool {
        if self.name != other.0 {
            return false;
        }
        match self.content {
            IndexingPositionContent::IndexerResult => {
                other.2 == HeadKind::Indexer && other.1 == usize::MAX
            }
            IndexingPositionContent::IndexerSpecificIndex(pos) => {
                other.2 == HeadKind::Indexer && other.1 == pos
            }
            IndexingPositionContent::TensorSingleIndex => {
                other.2 == HeadKind::Tensor && other.1 == 0
            }
            IndexingPositionContent::TensorSpecificIndex(pos) => {
                other.2 == HeadKind::Tensor && other.1 == pos
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Discriminant {
    data: String,
}

impl Discriminant {
    pub fn new() -> Self {
        Self {
            data: String::new(),
        }
    }

    pub fn extend(&self, i: usize) -> Self {
        let c = *Self::CHARS.get(i).expect("Too many indexes!");
        let data = format!("{}{}", &self.data, c);
        Self { data }
    }

    const CHARS: &[char] = &[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l',
    ];
}

impl Deref for Discriminant {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

pub struct IndexingPositionEquivalence {
    pub index: Option<(Ident, Discriminant)>,
    pub positions: Vec<IndexingPosition>,
}

impl IndexingPositionEquivalence {
    pub fn new(index: Ident, discriminant: Discriminant) -> Self {
        Self {
            index: Some((index, discriminant)),
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
