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
    name: Ident,
    position: usize,
    kind: HeadKind,
}

impl IndexingPosition {
    pub fn new(name: &Ident, position: usize, kind: HeadKind) -> Self {
        Self {
            name: name.clone(),
            position,
            kind,
        }
    }
    pub fn new_indexer_result(name: &Ident) -> Self {
        Self {
            name: name.clone(),
            position: Self::RETURN_POSITION,
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
    positions: Vec<IndexingPosition>,
}

impl IndexingPositionEquivalence {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
        }
    }
}

pub struct IndexingPositionMapping {
    equivalences: Vec<IndexingPositionEquivalence>,
    plain_values: HashMap<IndexingPosition, Expr>,
}

impl IndexingPositionMapping {
    pub fn new() -> Self {
        Self {
            equivalences: Vec::new(),
            plain_values: HashMap::new(),
        }
    }
    pub fn get_equivalences(&self) -> &[IndexingPositionEquivalence] {
        &self.equivalences
    }
    pub fn get_plain_values(&self) -> &HashMap<IndexingPosition, Expr> {
        &self.plain_values
    }
}

pub fn inspect(group: &RicciGroup) -> Result<IndexingPositionMapping, syn::Error> {
    let mut mapping = IndexingPositionMapping::new();
    let mut equivalences_per_index: HashMap<Ident, IndexingPositionEquivalence> = HashMap::new();
    inspect_group(group, &mut mapping, &mut equivalences_per_index)?;
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

fn inspect_indexers(
    head_name: &Ident,
    position: usize,
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
                        .push(IndexingPosition::new(head_name, position, kind))
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
                        .push(IndexingPosition::new(head_name, position, kind))
                });
        }
        RicciIndexer::Reindexing {
            reindexing_name,
            indexers,
        } => {
            let positions = vec![
                IndexingPosition::new(head_name, position, kind),
                IndexingPosition::new_indexer_result(reindexing_name),
            ];
            mapping
                .equivalences
                .push(IndexingPositionEquivalence { positions });
            for (position, indexer) in indexers.iter().enumerate() {
                inspect_indexers(
                    reindexing_name,
                    position,
                    HeadKind::Indexer,
                    indexer,
                    mapping,
                    equivalences_per_index,
                )?;
            }
        }
        RicciIndexer::Plain { expr } => {
            mapping.plain_values.insert(
                IndexingPosition::new(head_name, position, kind),
                *expr.clone(),
            );
        }
    }
    Ok(())
}

fn inspect_group(
    group: &RicciGroup,
    mapping: &mut IndexingPositionMapping,
    equivalences_per_index: &mut HashMap<Ident, IndexingPositionEquivalence>,
) -> Result<(), syn::Error> {
    for segment in &group.segments {
        match segment {
            RicciSegment::TensorCall {
                tensor_name,
                indexers,
            } => {
                for (position, indexer) in indexers.iter().enumerate() {
                    inspect_indexers(
                        tensor_name,
                        position,
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
                    equivalences_per_index
                        .insert(index.clone(), IndexingPositionEquivalence::new());
                }
                inspect_group(&lambda.body, mapping, equivalences_per_index)?;
                for index in new_indexes {
                    let equivalence = equivalences_per_index.remove(&index).unwrap();
                    if !equivalence.positions.is_empty() {
                        mapping.equivalences.push(equivalence);
                    }
                }
            }
            RicciSegment::SubGroup { group, .. } => {
                inspect_group(group, mapping, equivalences_per_index)?;
            }
            RicciSegment::Token(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        inspection::{HeadKind, inspect},
        model::lambda::RicciGroup,
    };

    use quote::quote;
    use syn::parse2;

    #[test]
    fn inspect_lambda() {
        let tokens = quote!(for i => a[i] + 3);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let equivalences = mapping.get_equivalences();
        let plain_values = mapping.get_plain_values();

        assert_eq!(0, plain_values.len());
        assert_eq!(1, equivalences.len());
        assert_eq!(1, equivalences[0].positions.len());
        assert_eq!(&equivalences[0].positions[0], &("a", 0, HeadKind::Tensor));

        let tokens = quote!(for i j k => a[i, j] + b[j, k] * c[i]);

        let lambda = parse2::<RicciGroup>(tokens).unwrap();

        let mapping = inspect(&lambda).unwrap();

        let equivalences = mapping.get_equivalences();
        let plain_values = mapping.get_plain_values();

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

        let equivalences = mapping.get_equivalences();

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

        let equivalences = mapping.get_equivalences();

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
