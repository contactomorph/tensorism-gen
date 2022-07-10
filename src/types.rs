use std::collections::HashMap;
use proc_macro2::{Ident, Span, TokenTree};

pub struct EinsteinFunction {
    pub span: Span,
    pub inverted_indexes: Vec<Ident>,
    pub content: Vec<EinsteinAlternative>,
}

impl EinsteinFunction {
    pub fn new(span: Span, inverted_indexes: Vec<Ident>) -> Self {
        EinsteinFunction {
            span,
            inverted_indexes,
            content: Vec::new(),
        }
    }

    pub fn from_group(span: Span, content: Vec<EinsteinAlternative>) -> Self {
        EinsteinFunction {
            span,
            inverted_indexes: Vec::new(),
            content,
        }
    }

    pub fn push_token(&mut self, token: TokenTree) {
        self.content.push(EinsteinAlternative::Tree(token));
    }

    pub fn retrieve_all_indexes(&mut self) -> Vec<Ident> {
        let mut inverted_indexes = Vec::<Ident>::new();
        while let Some(EinsteinAlternative::Tree(TokenTree::Ident(index))) = self.content.last() {
            inverted_indexes.push(index.clone());
            self.content.pop();
        }
        inverted_indexes
    }
}

pub struct EinsteinPosition {
    pub tensor_name: Ident,
    pub position: usize,
}

pub enum EinsteinAlternative {
    Func(EinsteinFunction),
    Tree(TokenTree),
    ParensGroup(Span, Vec<EinsteinAlternative>),
    TensorAccess {
        tensor_name: Ident,
        span: Span,
        indexes: Vec<Ident>,
    },
}

pub struct IndexUse {
    indexes_in_order: Vec<Ident>,
    correspondence: HashMap<String, Vec<EinsteinPosition>>,
}

impl IndexUse {
    pub fn new() -> Self {
        IndexUse {
            indexes_in_order: Vec::new(),
            correspondence: HashMap::new(),
        }
    }

    pub fn push(&mut self, index_name: Ident, tensor_name: Ident, position: usize) {
        let position = EinsteinPosition {
            tensor_name,
            position,
        };
        let index_as_string = index_name.to_string();
        match self.correspondence.get_mut(&index_as_string) {
            Some(positions) => {
                positions.push(position);
            }
            None => {
                let mut positions = Vec::new();
                positions.push(position);
                self.correspondence.insert(index_as_string, positions);
                self.indexes_in_order.push(index_name);
            }
        }
    }
    pub fn into_iter(self) -> impl IntoIterator<Item = (Ident, Vec<EinsteinPosition>)> {
        let mut correspondence = self.correspondence;
        self.indexes_in_order.into_iter().map(move |index_name| {
            let positions = correspondence.remove(&index_name.to_string()).unwrap();
            (index_name, positions)
        })
    }
}
