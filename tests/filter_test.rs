use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn filter_lambda_format() {
    let format = format_new_ndarray!(for i => (for j if tensor2[j] < j as i32 => tensor1[i, j]).sum::<i32>() + i as i32);

    asserts::equivalent!(
        format,
        r#"{
            let tsm_dim_1 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2);
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 1 {
                panic ! ("Dimensions are not matching between tensor2[j] and tensor1[_, j]");
            };
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 0 ;
            let tsm_ptr_tensor1 : * const _ = tensor1 . as_ptr ();
            let tsm_strides = tensor1 . strides ();
            let tsm_stride_0_tensor1 : isize = tsm_strides [0usize];
            let tsm_stride_1_tensor1 : isize = tsm_strides [1usize];
            let tsm_ptr_tensor2 : * const _ = tensor2 . as_ptr ();
            let tsm_strides = tensor2 . strides ();
            let tsm_stride_0_tensor2 : isize = tsm_strides [0usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {((0usize ..). map (| _ | {(* unsafe {& * tsm_ptr_tensor1})})). sum ::< i32 > ()+ 0usize as i32});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    ((0usize .. tsm_dim_1). filter (| & j | {(* unsafe {& * tsm_ptr_tensor2 . offset ((j as isize)* tsm_stride_0_tensor2)})< j as i32}). map (| j | {(* unsafe {& * tsm_ptr_tensor1 . offset ((i as isize)* tsm_stride_0_tensor1 + (j as isize)* tsm_stride_1_tensor1)})})). sum ::< i32 > ()+ i as i32
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "#
    );

    let format = format_new_ndarray!(for i => Iterator::min(for j k if tensor3[j] < tensor4[k] => tensor1[i, j] * tensor2[i, k]));

    asserts::equivalent!(
        format,
        r##"{
            let tsm_dim_1 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor3);
            if tsm_dim_1 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 1 {
                panic ! ("Dimensions are not matching between tensor3[j] and tensor1[_, j]");
            };
            let tsm_dim_2 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor4);
            if tsm_dim_2 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2). 1 {
                panic ! ("Dimensions are not matching between tensor4[k] and tensor2[_, k]");
            };
            let tsm_dim_0 = :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor1). 0 ;
            if tsm_dim_0 != :: ndarray :: ArrayBase :: < _ , _ > :: dim (& tensor2). 0 {
                panic ! ("Dimensions are not matching between tensor1[i, _] and tensor2[i, _]");
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
            let tsm_ptr_tensor4 : * const _ = tensor4 . as_ptr ();
            let tsm_strides = tensor4 . strides ();
            let tsm_stride_0_tensor4 : isize = tsm_strides [0usize];
            type TsmDimensionType = :: ndarray :: Dim < [:: ndarray :: Ix ; 1usize]> ;
            let mut tsm_res = :: ndarray :: Array :: < _ , TsmDimensionType > :: uninit ((tsm_dim_0 ,));
            let mut tsm_res_ptr = tsm_res . as_mut_ptr ()as * mut _ ;
            fn tsm_unify < T , D > (_tensor : & :: ndarray :: Array :: < std :: mem :: MaybeUninit < T > , D > , _ptr : * mut T , _f : impl Fn ()-> T ,){}
            tsm_unify (& tsm_res , tsm_res_ptr , || {Iterator :: min ((0usize ..). map (| _ | {(* unsafe {& * tsm_ptr_tensor1})* (* unsafe {& * tsm_ptr_tensor2})}))});
            for i in 0usize .. tsm_dim_0 {
                let tsm_value = {
                    Iterator :: min ((0usize .. tsm_dim_2). flat_map (move | k | {(0usize .. tsm_dim_1). map (move | j | {(j , k ,)})}). filter (| & (j , k ,)| {(* unsafe {& * tsm_ptr_tensor3 . offset ((j as isize)* tsm_stride_0_tensor3)})< (* unsafe {& * tsm_ptr_tensor4 . offset ((k as isize)* tsm_stride_0_tensor4)})}). map (| (j , k ,)| {(* unsafe {& * tsm_ptr_tensor1 . offset ((i as isize)* tsm_stride_0_tensor1 + (j as isize)* tsm_stride_1_tensor1)})* (* unsafe {& * tsm_ptr_tensor2 . offset ((i as isize)* tsm_stride_0_tensor2 + (k as isize)* tsm_stride_1_tensor2)})}))
                };
                unsafe {tsm_res_ptr . write (tsm_value); tsm_res_ptr = tsm_res_ptr . add (1);}
            };
            unsafe {tsm_res . assume_init ()}
        } "##
    );
}

#[test]
fn filter_lambda_generation() {
    let tensor1 = Array2::<i32>::from_shape_fn((3, 5), |(i, j)| (i as i32) * (j as i32) - 7);
    let tensor2 = Array1::<i32>::from_shape_fn(5, |i| 1 - (i as i32));
    let result = new_ndarray!(for i => (for j if tensor2[j] < j as i32 => tensor1[i, j]).sum::<i32>() + i as i32);
    let expected = Array1::<i32>::from_shape_fn(3, |i| {
        (0..5)
            .filter(|&j| tensor2[j] < j as i32)
            .map(|j| tensor1[(i, j)])
            .sum::<i32>()
            + i as i32
    });
    assert_eq!(expected, result);

    let tensor1 = Array2::<f64>::from_shape_fn((3, 5), |(i, j)| (i as f64) * (j as f64) - 1.5);
    let tensor2 = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| 0.5 * (j as f64) - (i as f64));
    let tensor3 = Array1::<f64>::from_shape_fn(5, |i| i as f64 - 2.0);
    let tensor4 = Array1::<f64>::from_shape_fn(2, |i| 1.0 - (i as f64));
    let result = new_ndarray!(for i => (for j k if tensor3[j] < tensor4[k] => tensor1[i, j] * tensor2[i, k]).sum::<f64>());
    let expected = Array1::<f64>::from_shape_fn(3, |i| {
        (0..2)
            .flat_map(|k| (0..5).map(move |j| (j, k)))
            .filter(|&(j, k)| tensor3[j] < tensor4[k])
            .map(|(j, k)| tensor1[(i, j)] * tensor2[(i, k)])
            .sum::<f64>()
    });
    assert_eq!(expected, result);
}
