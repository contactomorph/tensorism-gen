use crate::model::lambda::{RicciGroup, RicciLambda};

pub enum TopGroup {
    Group(RicciGroup),
    Lambda(Box<RicciLambda>),
}
