extern crate proc_macro;
extern crate syn;
//#[macro_use]
//extern crate quote;
use core::iter::FromIterator;
use proc_macro::{TokenStream, TokenTree, Delimiter, Ident, Span};

struct EinsteinExpression {
    delimiter: Delimiter,
    span: Span,
    indexes: Vec::<Ident>,
    content: Vec::<EinsteinAlternative>,
}

impl EinsteinExpression {
    fn new(span: Span) -> Self {
        EinsteinExpression {
            delimiter: Delimiter::None,
            span,
            indexes: Vec::new(),
            content: Vec::new(),
        }
    }
}

enum EinsteinSubscript {
    Index(Ident),
    ComplexExpression(Vec<TokenTree>),
}

enum EinsteinAlternative {
    EinsteinSubExp(EinsteinExpression),
    TokenTree(TokenTree),
    TensorSubscripting { name: TokenTree, indexes: Vec<EinsteinSubscript> },
}

fn to_einstein_expression(tokens: &mut impl Iterator<Item=TokenTree>, exp: &mut EinsteinExpression) {
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(ref p) => {
                match p.as_char() {
                    ';' => {
                        panic!("Statements are forbidden");
                    },
                    '$' => {
                        for _ in 0..exp.indexes.len() { exp.content.pop(); }
                        let mut new_einst = EinsteinExpression::new(p.span());
                        to_einstein_expression(tokens, &mut new_einst);
                        exp.content.push(EinsteinAlternative::EinsteinSubExp(new_einst));
                    },
                    _ => {
                        exp.content.push(EinsteinAlternative::TokenTree(token));
                    },
                }
            },
            TokenTree::Ident(ref indent) => {
                exp.indexes.push(indent.clone());
                exp.content.push(EinsteinAlternative::TokenTree(token));
            },
            TokenTree::Group(group) => {
                match group.delimiter() {
                    Delimiter::Bracket => {

                    },
                    _ => {

                    },
                }
            },
            TokenTree::Literal(_) => {
                exp.indexes.clear();
                exp.content.push(EinsteinAlternative::TokenTree(token));
            },
        }
    }
}

fn from_einstein_expression(exp: &EinsteinExpression, tokens: &mut Vec<TokenTree>) {
    for alt in exp.content.iter() {
        match alt {
            EinsteinAlternative::EinsteinSubExp(ref EinsteinSubExp) =>
                from_einstein_expression(EinsteinSubExp, tokens),
            EinsteinAlternative::TokenTree(ref token) =>
                tokens.push(token.clone()),
            //EinsteinAlternative::TensorSubscripting { .. } => {},
            _ => {},
        }
    }
}

#[proc_macro]
pub fn decl(s: TokenStream) -> TokenStream {
    let mut exp = EinsteinExpression::new(Span::call_site());
    to_einstein_expression(&mut s.into_iter(), &mut exp);
    let mut tokens = Vec::new();
    from_einstein_expression(&exp, &mut tokens);
    FromIterator::<TokenTree>::from_iter(tokens.into_iter())
}
