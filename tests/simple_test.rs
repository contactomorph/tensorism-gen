use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn simple_lambda_format() {
    let format = format_new_ndarray!(for i => tensor[i] + i);

    asserts::equivalent!(
        format,
        r"{
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor);
            let tsm_ptr_tensor : * const _ = tensor . as_ptr ();
            let tsm_strides = tensor . strides ();
            let tsm_stride_0_tensor : isize = tsm_strides [0usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor})+ 0usize});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    (* unsafe {& * tsm_ptr_tensor . offset ((i as isize)* tsm_stride_0_tensor)})+ i
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "
    );

    let format = format_new_ndarray!(for i j => tensor[i, j]);

    asserts::equivalent!(
        format,
        r"{
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor). 0 ;
            let tsm_dim_1 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor). 1 ;
            let tsm_ptr_tensor : * const _ = tensor . as_ptr ();
            let tsm_strides = tensor . strides ();
            let tsm_stride_0_tensor : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor : isize = tsm_strides [1usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 2usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 , tsm_dim_1 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {(* unsafe {& * tsm_ptr_tensor})});
            for i in 0usize .. tsm_dim_0 {
                for j in 0usize .. tsm_dim_1 {
                    let tsm_value = {
                        (* unsafe {& * tsm_ptr_tensor . offset ((i as isize)* tsm_stride_0_tensor + (j as isize)* tsm_stride_1_tensor)})
                    };
                    unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
                };
            };
            unsafe {tsm_res . assume_init ()}
        } "
    );
}

#[test]
fn simple_lambda_generation() {
    let tensor = Array1::<f64>::from_shape_fn(13, |i| 20.0 - i as f64 * 1.5);
    let modified_tensor = new_ndarray!(for i => tensor[i] + i as f64);
    let mut i = 0;
    let i = &mut i;
    let expected_tensor = tensor.map(|x| {
        let v = x + *i as f64;
        *i += 1;
        v
    });
    assert_eq!(expected_tensor, modified_tensor);

    let tensor = Array2::<f64>::from_shape_fn((10, 9), |(i, j)| i as f64 * 3.0 + j as f64 * 7.5);
    let transpose = new_ndarray!(for i j => tensor[j, i]);
    assert_eq!(tensor.reversed_axes(), transpose);
}

#[test]
fn layered_lambda_format() {
    let format = format_new_ndarray!(for i => (for j => tensor1[i, j]).sum::<i32>() + (for j => tensor2[j, i]).sum::<i32>() * tensor3[i]);
    asserts::equivalent!(
        format,
        r##"{
            let tsm_dim_1 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 1 ;
            let tsm_dim_2 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2). 0 ;
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 0 ;
            if tsm_dim_0 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2). 1 {
                panic ! ("Dimensions are not matching between tensor1[i, _] and tensor2[_, i]");
            };
            if tsm_dim_0 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3){
                panic ! ("Dimensions are not matching between tensor1[i, _] and tensor3[i]");
            };
            let tsm_ptr_tensor1 : * const _ = tensor1 . as_ptr ();
            let tsm_strides = tensor1 . strides ();
            let tsm_stride_0_tensor1 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor1 : isize = tsm_strides [1usize];
            let tsm_ptr_tensor2 : * const _ = tensor2 . as_ptr ();
            let tsm_strides = tensor2 . strides ();
            let tsm_stride_0_tensor2 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor2 : isize = tsm_strides [1usize];
            let tsm_ptr_tensor3 : * const _ = tensor3 . as_ptr ();
            let tsm_strides = tensor3 . strides ();
            let tsm_stride_0_tensor3 : isize = tsm_strides [0usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {((0usize ..). map (| _ | {(* unsafe {& * tsm_ptr_tensor1})})). sum ::< i32 > ()+ ((0usize ..). map (| _ | {(* unsafe {& * tsm_ptr_tensor2})})). sum ::< i32 > ()* (* unsafe {& * tsm_ptr_tensor3})});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    ((0usize .. tsm_dim_1). map (| j | {(* unsafe {& * tsm_ptr_tensor1 . offset ((i as isize)* tsm_stride_0_tensor1 + (j as isize)* tsm_stride_1_tensor1)})})). sum ::< i32 > ()+ ((0usize .. tsm_dim_2). map (| j | {(* unsafe {& * tsm_ptr_tensor2 . offset ((j as isize)* tsm_stride_0_tensor2 + (i as isize)* tsm_stride_1_tensor2)})})). sum ::< i32 > ()* (* unsafe {& * tsm_ptr_tensor3 . offset ((i as isize)* tsm_stride_0_tensor3)})
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "##
    );
}

#[test]
fn layered_lambda_generation() {
    let tensor1 = Array2::<i32>::from_shape_fn((7, 2), |(i, j)| {
        ((i * j) as i32 - 3 * (i as i32) + 5 * (j as i32) - 6) % 11
    });
    let tensor2 = Array2::<i32>::from_shape_fn((2, 7), |(i, j)| {
        ((i * j) as i32 + 3 * (i as i32) - 5 * (j as i32) - 6) % 11
    });
    let tensor3 = Array1::<i32>::from_shape_fn(7, |i| ((3 * i + 4) % 7) as i32);
    let result = new_ndarray!(for i => (for j => tensor1[i, j]).sum::<i32>() + (for j => tensor2[j, i]).sum::<i32>() * tensor3[i]);
    let expected = Array1::from_vec(vec![-43i32, -12, -21, -95, -40, -60, -12]);
    assert_eq!(expected, result)
}

#[test]
fn tensor_references_are_accepted() {
    let mut tensor1 = Array1::<i32>::from_shape_fn(7, |i| i as i32);
    let tensor2 = Array1::<i32>::from_shape_fn(7, |i| -(i as i32));

    let tensor1_ref = &mut tensor1;
    let tensor2_ref = &tensor2;

    let result = new_ndarray!(for i => tensor1_ref[i] + tensor2_ref[i]);
    let expected = Array1::from_vec(vec![0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(expected, result)
}

#[test]
#[should_panic(expected = "Dimensions are not matching between tensor1[i] and tensor2[i]")]
fn panic_when_dimensions_are_non_matching() {
    let tensor1 = Array1::<i32>::from_shape_fn(7, |i| i as i32);
    let tensor2 = Array1::<i32>::from_shape_fn(6, |i| -(i as i32));
    new_ndarray!(for i => tensor1[i] + tensor2[i]);
}
