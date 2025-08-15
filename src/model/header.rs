use std::fmt::Display;

use quote::ToTokens;
use syn::Error;
use syn::bracketed;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Result, Token};

pub struct RicciIndexDeclaration {
    pub indexes: Vec<Ident>,
}

pub enum RicciIndexer {
    Direct {
        index: Ident,
    },
    Reindexing {
        reindexing_name: Ident,
        indexers: Vec<RicciIndexer>,
    },
    Reverse {
        index: Ident,
    },
    Plain {
        expr: Box<Expr>,
    },
}

pub struct RicciAliasDeclaration {
    pub index: Ident,
    pub indexer: RicciIndexer,
}

impl Parse for RicciIndexDeclaration {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<Token![for]>()?;

        let mut indexes: Vec<Ident> = Vec::new();
        let index: Ident = input.parse()?;
        indexes.push(index);

        while input.peek(syn::Ident) {
            let ident: Ident = input.parse()?;
            indexes.push(ident);
        }

        Ok(Self { indexes })
    }
}

fn parse_keyword_reindexing(keyword_token: Ident, input: ParseStream) -> Result<RicciIndexer> {
    if keyword_token == "plain" {
        let expr: Box<Expr> = input.parse()?;
        Ok(RicciIndexer::Plain { expr })
    } else if keyword_token == "rev" {
        let index: Ident = input.parse()?;
        Ok(RicciIndexer::Reverse { index })
    } else {
        Err(Error::new(
            keyword_token.span(),
            format!("Unknown reindexing keyword: {}", keyword_token),
        ))
    }
}

fn parse_external_reindexing(reindexing_name: Ident, input: ParseStream) -> Result<RicciIndexer> {
    let content;
    bracketed!(content in input);
    let mut indexers: Vec<RicciIndexer> = Vec::new();
    let mut first = true;
    while !content.is_empty() {
        if first {
            first = false;
        } else {
            content.parse::<Token![,]>()?;
        }
        let indexer: RicciIndexer = content.parse()?;
        indexers.push(indexer);
    }
    Ok(RicciIndexer::Reindexing {
        reindexing_name,
        indexers,
    })
}

impl Parse for RicciIndexer {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(syn::Ident) {
            let keyword_token: Ident = input.parse()?;
            if input.is_empty() {
                Ok(RicciIndexer::Direct {
                    index: keyword_token,
                })
            } else if input.peek(Token![:]) {
                input.parse::<Token![:]>()?;
                parse_keyword_reindexing(keyword_token, input)
            } else if input.peek(syn::token::Bracket) {
                parse_external_reindexing(keyword_token, input)
            } else {
                Ok(RicciIndexer::Direct {
                    index: keyword_token,
                })
            }
        } else {
            Err(Error::new(
                input.span(),
                "Expected an identifier for reindexing",
            ))
        }
    }
}

impl Parse for RicciAliasDeclaration {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<Token![let]>()?;
        let index: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let reindexer: RicciIndexer = input.parse()?;

        Ok(Self {
            index,
            indexer: reindexer,
        })
    }
}

impl Display for RicciIndexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RicciIndexer::Direct { index } => write!(f, " {}", index),
            RicciIndexer::Reindexing {
                reindexing_name,
                indexers,
            } => {
                f.write_fmt(format_args!(" {} ⦇", reindexing_name))?;
                let mut first = true;
                for indexer in indexers {
                    if !first {
                        f.write_str(" ,")?;
                    }
                    first = false;
                    indexer.fmt(f)?;
                }
                f.write_str(" ⦈")
            }
            RicciIndexer::Reverse { index } => f.write_fmt(format_args!(" ↺ {}", index)),
            RicciIndexer::Plain { expr } => {
                f.write_str(" « ")?;
                expr.to_token_stream().fmt(f)?;
                f.write_str(" »")
            }
        }
    }
}

impl Display for RicciIndexDeclaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("∀")?;
        for index in &self.indexes {
            f.write_fmt(format_args!(" {}", index))?;
        }
        Ok(())
    }
}

impl Display for RicciAliasDeclaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("∙ {} ≔{}", self.index, self.indexer))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert::Assert,
        model::header::{RicciAliasDeclaration, RicciIndexDeclaration},
    };

    use quote::quote;

    #[test]
    fn parse_header() {
        let tokens = quote!(for i j);

        assert_eq!(
            Assert::parse_and_display::<RicciIndexDeclaration>(tokens),
            "∀ i j"
        );

        let tokens = quote!(let i = plain: 4 + g());

        assert_eq!(
            Assert::parse_and_display::<RicciAliasDeclaration>(tokens),
            "∙ i ≔ « 4 + g () »"
        );

        let tokens = quote!(let i = rev: long_index_name);

        assert_eq!(
            Assert::parse_and_display::<RicciAliasDeclaration>(tokens),
            "∙ i ≔ ↺ long_index_name"
        );

        let tokens = quote!(let i = f[a, b, c]);

        assert_eq!(
            Assert::parse_and_display::<RicciAliasDeclaration>(tokens),
            "∙ i ≔ f ⦇ a , b , c ⦈"
        );
    }
}
