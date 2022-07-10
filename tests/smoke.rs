use std::str::FromStr;

use tensorism::{shapes::ShapeBuilder, tensors::TensorBuilder};
use tensorism_gen::{tensorism_make, tensorism_string_for_make};

#[test]
fn format_make_macro() {
    let string = tensorism_string_for_make! {(i j $ a[i, j] + i as f64).sum()};
    assert_eq!(
        "{\n    \
            let i_length : usize = :: tensorism :: tensors :: Tensor ::\n    dims(& a).0.into() ; \
            let j_length : usize = :: tensorism :: tensors ::\n    Tensor :: dims(& a).1.into() ;\n    \
            ((0usize ..\n    i_length).flat_map(move | i |\n    {\n        (0usize .. j_length).map(move | j | { (i, j,) })\n    })\
            .map(| (i, j,) |\n    { (* unsafe { a.get_unchecked(i, j) }) + i as f64 })).sum()\n\
        }",
        string);
    let string = tensorism_string_for_make! {i $ (j $ a[i, j] + b[j])};
    assert_eq!(
        "{\n    \
            let i_length : usize = :: tensorism :: tensors :: Tensor ::\n    dims(& a).0.into() ; \
            let j_length : usize = :: tensorism :: tensors ::\n    Tensor :: dims(& a).1.into() ;\n    \
            (0usize ..\n    i_length).map(move | i |\n    {\n        (i,)\n    }).map(| (i,) |\n    {\n        ((0usize ..\n        j_length).map(move | j |\n        {\n            (j,)\n        })\
            .map(| (j,) |\n        {\n            (* unsafe { a.get_unchecked(i, j) }) +\n            (* unsafe { b.get_unchecked(j) })\n        }))\n    })\n\
        }",
        string);
}

fn count_all_chars<'a>(it: impl Iterator<Item = &'a String>) -> usize {
    it.fold(0usize, |acc, message| acc + message.chars().count())
}

#[test]
fn run_make_macro() {
    let a = ShapeBuilder::with_static::<10>()
        .with_first()
        .prepare()
        .fill(&456i64);
    let sum: i64 = tensorism_make! {(i j $ a[i, j] + i as i64).sum()};
    assert_eq!(46050i64, sum);

    let result: i64 = tensorism_make! {(i $ (j $ a[i, j]).min().unwrap()).sum()};
    assert_eq!(4560i64, result);

    let mut messages = ["Hello", "World", "How", "are you?"]
        .map(|s| String::from_str(s).unwrap())
        .to_vec();
    let c = ShapeBuilder::with_static::<4>()
        .prepare()
        .append(&mut messages)
        .generate();
    let all_chars_count = tensorism_make! {count_all_chars(i $ &c[i])};
    assert_eq!(21, all_chars_count);
    // let b = ShapeBuilder::with_static::<10>()
    //     .prepare()
    //     .fill(&12f64);
    // let iterator = tensorism_make! {i $ (j $ a[i, j] + b[j])};
}

//let v = tensorism_gen::decl!(i # a[i, 4] + b[i]);
//let v = {
//  if a.size1 != b.size1 { panic!("") };
//  (1..a.size1).build_iter_1(|i| a.get_raw(i, 4) + b.get_raw(i)).to_tensor_2();
//};

// #[fail]
// fn do_not_works() {
//     let value = tensorism_gen::decl!{
//         j $ i;
//     };
// }
