use std::collections::HashMap;

use proc_macro2::Ident;
use syn::Expr;

use crate::model::{
    header::RicciIndexer,
    lambda::{RicciGroup, RicciSegment},
};

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

pub fn inspect(group: &RicciGroup) -> Result<IndexingPositionMapping, syn::Error> {
    let postfix = String::new();
    let mut mapping = IndexingPositionMapping::new();
    let mut equivalences_per_index: HashMap<Ident, IndexingPositionEquivalence> = HashMap::new();
    inspect_group(group, postfix, &mut mapping, &mut equivalences_per_index)?;
    Ok(mapping)
}

fn create_unknown_index_error(index: &Ident) -> Result<(), syn::Error> {
    Err(syn::Error::new_spanned(
        index,
        format!("Index '{}' is not declared in the current scope", index),
    ))
}

fn create_duplicated_index_error(index: &Ident) -> Result<(), syn::Error> {
    Err(syn::Error::new_spanned(
        index,
        format!("Index '{}' is already declared in the current scope", index),
    ))
}

const POSTFIX_CHARS: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j',
];

fn to_char(i: usize) -> char {
    *POSTFIX_CHARS.get(i).expect("Too many indexes!")
}

fn inspect_indexers(
    head_name: &Ident,
    position: usize,
    rank: usize,
    kind: HeadKind,
    indexer: &RicciIndexer,
    mapping: &mut IndexingPositionMapping,
    equivalences_per_index: &mut HashMap<Ident, IndexingPositionEquivalence>,
) -> Result<(), syn::Error> {
    match indexer {
        RicciIndexer::Direct { index } => {
            if !equivalences_per_index.contains_key(index) {
                create_unknown_index_error(index)?;
            }
            equivalences_per_index
                .entry(index.clone())
                .and_modify(|eq| {
                    eq.positions
                        .push(IndexingPosition::new(head_name, position, rank, kind))
                });
        }
        RicciIndexer::Reverse { index } => {
            if !equivalences_per_index.contains_key(index) {
                create_unknown_index_error(index)?;
            }
            equivalences_per_index
                .entry(index.clone())
                .and_modify(|eq| {
                    eq.positions
                        .push(IndexingPosition::new(head_name, position, rank, kind))
                });
        }
        RicciIndexer::Reindexing {
            reindexing_name,
            indexers,
        } => {
            let position_a = IndexingPosition::new(head_name, position, rank, kind);
            let position_b = IndexingPosition::new_indexer_result(reindexing_name);
            let equivalence = IndexingPositionEquivalence::from_positions(position_a, position_b);
            mapping.equivalences.push(equivalence);
            let rank = indexers.len();
            for (position, indexer) in indexers.iter().enumerate() {
                inspect_indexers(
                    reindexing_name,
                    position,
                    rank,
                    HeadKind::Indexer,
                    indexer,
                    mapping,
                    equivalences_per_index,
                )?;
            }
        }
        RicciIndexer::Plain { expr } => {
            mapping.plain_values.insert(
                IndexingPosition::new(head_name, position, rank, kind),
                *expr.clone(),
            );
        }
    }
    Ok(())
}

fn inspect_group(
    group: &RicciGroup,
    postfix: String,
    mapping: &mut IndexingPositionMapping,
    equivalences_per_index: &mut HashMap<Ident, IndexingPositionEquivalence>,
) -> Result<(), syn::Error> {
    for (i, segment) in group.segments.iter().enumerate() {
        match segment {
            RicciSegment::TensorCall {
                tensor_name,
                indexers,
            } => {
                let rank = indexers.len();
                for (position, indexer) in indexers.iter().enumerate() {
                    inspect_indexers(
                        tensor_name,
                        position,
                        rank,
                        HeadKind::Tensor,
                        indexer,
                        mapping,
                        equivalences_per_index,
                    )?;
                }
            }
            RicciSegment::SubLambda(lambda) => {
                let mut new_indexes = Vec::new();
                for index in &lambda.index_declaration.indexes {
                    if equivalences_per_index.contains_key(index) {
                        create_duplicated_index_error(index)?;
                    }
                    new_indexes.push(index.clone());
                    let equivalence =
                        IndexingPositionEquivalence::new(index.clone(), postfix.clone());
                    equivalences_per_index.insert(index.clone(), equivalence);
                }
                let new_postfix = format!("{}{}", postfix, to_char(i));
                inspect_group(&lambda.body, new_postfix, mapping, equivalences_per_index)?;
                for index in new_indexes {
                    let equivalence = equivalences_per_index.remove(&index).unwrap();
                    if !equivalence.positions.is_empty() {
                        mapping.equivalences.push(equivalence);
                    }
                }
            }
            RicciSegment::SubGroup { group, .. } => {
                let new_postfix = format!("{}{}", postfix, to_char(i));
                inspect_group(group, new_postfix, mapping, equivalences_per_index)?;
            }
            RicciSegment::Token(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        inspection::{HeadKind, IndexingPositionMapping, inspect},
        model::lambda::RicciGroup,
    };

    use quote::quote;
    use syn::parse2;

    #[test]
    fn inspect_lambda() {
        let tokens = quote!(for i => a[i] + 3);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping {
            equivalences,
            plain_values,
        } = mapping;

        assert_eq!(0, plain_values.len());
        assert_eq!(1, equivalences.len());
        assert_eq!(1, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("a", 0, HeadKind::Tensor));

        let tokens = quote!(for i j k => a[i, j] + b[j, k] * c[i]);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping {
            equivalences,
            plain_values,
        } = mapping;

        assert_eq!(0, plain_values.len());
        assert_eq!(3, equivalences.len());
        assert_eq!(2, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("a", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[0].positions[1], &("c", 0, HeadKind::Tensor));
        assert_eq!(2, equivalences[1].positions.len());
        assert_eq!(&equivalences[1].positions[0], &("a", 1, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[1], &("b", 0, HeadKind::Tensor));
        assert_eq!(1, equivalences[2].positions.len());
        assert_eq!(&equivalences[2].positions[0], &("b", 1, HeadKind::Tensor));
    }

    #[test]
    fn inspect_group() {
        let tokens = quote!(product(for i => a[i] + sum(for j => b[i, j]) * mean(for j => c[j, i, j] / d[j])));

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping { equivalences, .. } = mapping;

        assert_eq!(3, equivalences.len());
        assert_eq!(1, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("b", 1, HeadKind::Tensor));
        assert_eq!(3, equivalences[1].positions.len());
        assert_eq!(&equivalences[1].positions[0], &("c", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[1], &("c", 2, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[2], &("d", 0, HeadKind::Tensor));
        assert_eq!(3, equivalences[2].positions.len());
        assert_eq!(&equivalences[2].positions[0], &("a", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[2].positions[1], &("b", 0, HeadKind::Tensor));
        assert_eq!(&equivalences[2].positions[2], &("c", 1, HeadKind::Tensor));

        let tokens = quote!(3.5 * median(
            for i => a[i] +
                sum(for j => if i < j { b[every3[i], j] } else { c[j] + 4.0 }) -
                max(for j => d[j, every4[j]] - e[sorted_by_a[j], i])
        ));

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let IndexingPositionMapping { equivalences, .. } = mapping;

        assert_eq!(6, equivalences.len());
        assert_eq!(2, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("b", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[0].positions[1],
            &("every3", usize::MAX, HeadKind::Indexer)
        );
        assert_eq!(2, equivalences[1].positions.len());
        assert_eq!(&equivalences[1].positions[0], &("b", 1, HeadKind::Tensor));
        assert_eq!(&equivalences[1].positions[1], &("c", 0, HeadKind::Tensor));
        assert_eq!(2, equivalences[2].positions.len());
        assert_eq!(&equivalences[2].positions[0], &("d", 1, HeadKind::Tensor));
        assert_eq!(
            &equivalences[2].positions[1],
            &("every4", usize::MAX, HeadKind::Indexer)
        );
        assert_eq!(2, equivalences[3].positions.len());
        assert_eq!(&equivalences[3].positions[0], &("e", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[3].positions[1],
            &("sorted_by_a", usize::MAX, HeadKind::Indexer)
        );
        assert_eq!(3, equivalences[4].positions.len());
        assert_eq!(&equivalences[4].positions[0], &("d", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[4].positions[1],
            &("every4", 0, HeadKind::Indexer)
        );
        assert_eq!(
            &equivalences[4].positions[2],
            &("sorted_by_a", 0, HeadKind::Indexer)
        );
        assert_eq!(3, equivalences[5].positions.len());
        assert_eq!(&equivalences[5].positions[0], &("a", 0, HeadKind::Tensor));
        assert_eq!(
            &equivalences[5].positions[1],
            &("every3", 0, HeadKind::Indexer)
        );
        assert_eq!(&equivalences[5].positions[2], &("e", 1, HeadKind::Tensor));
    }
}
