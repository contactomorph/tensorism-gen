use proc_macro2::{Ident, Span, TokenTree};
use std::collections::HashMap;

pub struct EinsteinFunction {
    pub inverted_indexes: Vec<Ident>,
    pub sequence: EinsteinSequence,
}

impl EinsteinFunction {
    pub fn new(inverted_indexes: Vec<Ident>, sequence: EinsteinSequence) -> Self {
        EinsteinFunction {
            sequence,
            inverted_indexes,
        }
    }
}

pub struct EinsteinPosition {
    pub tensor_name: Ident,
    pub position: usize,
    pub index_name: Ident,
}

pub enum EinsteinAlternative {
    Func(EinsteinFunction),
    Tree(TokenTree),
    Seq(EinsteinSequence),
    TensorAccess {
        tensor_name: Ident,
        span: Span,
        indexes: Vec<Ident>,
    },
}

pub struct EinsteinSequence {
    pub span: Span,
    pub use_parens: bool,
    pub content: Vec<EinsteinAlternative>,
}

impl EinsteinSequence {
    pub fn initial() -> Self {
        Self {
            span: Span::call_site(),
            use_parens: false,
            content: Vec::new(),
        }
    }

    pub fn with_parens(span: Span) -> Self {
        Self {
            span,
            use_parens: true,
            content: Vec::new(),
        }
    }

    pub fn naked(span: Span) -> Self {
        Self {
            span,
            use_parens: false,
            content: Vec::new(),
        }
    }

    pub fn push_token(&mut self, token: TokenTree) {
        self.content.push(EinsteinAlternative::Tree(token));
    }

    pub fn extract_previous_identifiers(&mut self) -> Vec<Ident> {
        let mut inverted_identifiers = Vec::<Ident>::new();
        while let Some(EinsteinAlternative::Tree(TokenTree::Ident(ident))) = self.content.last() {
            inverted_identifiers.push(ident.clone());
            self.content.pop();
        }
        inverted_identifiers
    }
}

pub struct IndexUse {
    indexes_in_order: Vec<String>,
    correspondence: HashMap<String, Vec<EinsteinPosition>>,
}

impl IndexUse {
    pub fn new() -> Self {
        IndexUse {
            indexes_in_order: Vec::new(),
            correspondence: HashMap::new(),
        }
    }

    pub fn declare_new(&mut self, index_name: Ident) -> bool {
        let index_as_string = index_name.to_string();
        if self.indexes_in_order.contains(&index_as_string) {
            false
        } else {
            self.indexes_in_order.push(index_as_string);
            true
        }
    }

    pub fn push(&mut self, index_name: Ident, tensor_name: Ident, position: usize) -> bool {
        let index_as_string = index_name.to_string();
        let position = EinsteinPosition {
            tensor_name,
            position,
            index_name: index_name.clone(),
        };
        match self.correspondence.get_mut(&index_as_string) {
            Some(positions) => {
                positions.push(position);
                true
            }
            None => {
                if self.indexes_in_order.contains(&&index_as_string) {
                    let mut positions = Vec::new();
                    positions.push(position);
                    self.correspondence.insert(index_as_string, positions);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn into_iter(self) -> impl IntoIterator<Item = (String, Vec<EinsteinPosition>)> {
        let mut correspondence = self.correspondence;
        self.indexes_in_order.into_iter().map(move |index_name| {
            let positions = correspondence.remove(&index_name.to_string()).unwrap();
            (index_name, positions)
        })
    }
}
