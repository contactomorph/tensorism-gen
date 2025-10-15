use tensorism_gen::format_new_ndarray;

#[test]
fn indexing_alias_format() {
    let format = format_new_ndarray!(for i let j = sort[i] => tensor1[j]);

    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = :: crate :: tensorism :: Reindexing1 :: get_input0_bound( & sort );
            let dim_number_1 = :: crate :: tensorism :: Reindexing1 :: get_output_bound( & sort );
            if dim_number_1 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor1 ) {
                panic! ( "Dimensions are not matching between j = sort[ _ ] and tensor1[ j ]" );
            }
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(
                dim_number_0,
                | i | {
                    let j = crate :: tensorism :: Reindexing1 :: get_unchecked( & sort, i );
                    (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor1, j) })
                }
            )
        } "#
    );

    let format = format_new_ndarray!(for i j let k = combine[i, j] => tensor2[i, k]);
    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = :: crate :: tensorism :: Reindexing2 :: get_input0_bound( & combine );
            if dim_number_0 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor2 ).0 {
                panic! ( "Dimensions are not matching between combine[ i, _ ] and tensor2[ i, _ ]" );
            }
            let dim_number_1 = :: crate :: tensorism :: Reindexing2 :: get_input1_bound( & combine );
            let dim_number_2 = :: crate :: tensorism :: Reindexing2 :: get_output_bound( & combine );
            if dim_number_2 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor2 ).1 {
                panic! ( "Dimensions are not matching between k = combine[ _, _ ] and tensor2[ _, k ]" );
            }
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 2usize] >> :: from_shape_fn(
                ( dim_number_0, dim_number_1, ),
                | ( i, j, ) | {
                    let k = crate :: tensorism :: Reindexing2 :: get_unchecked( & combine, i, j );
                    (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor2, (i, k, )) })
                }
            )
        } "#
    );

    let format =
        format_new_ndarray!(for i j let k = sup[middle[sub[i, j], j]] => tensor3[k, j, i]);
    asserts::equivalent!(
        format,
        r#"{
            let dim_number_3 = :: crate :: tensorism :: Reindexing1 :: get_input0_bound( & sup );
            if dim_number_3 != :: crate :: tensorism :: Reindexing1 :: get_output_bound( & middle ) {
                panic! ( "Dimensions are not matching between sup[ middle ] and middle = middle[ _ ]" );
            }
            let dim_number_4 = :: crate :: tensorism :: Reindexing2 :: get_input0_bound( & middle );
            if dim_number_4 != :: crate :: tensorism :: Reindexing2 :: get_output_bound( & sub ) {
                panic! ( "Dimensions are not matching between middle[ sub, _ ] and sub = sub[ _, _ ]" );
            }
            let dim_number_0 = :: crate :: tensorism :: Reindexing2 :: get_input0_bound( & sub );
            if dim_number_0 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor3 ).2 {
                panic! ( "Dimensions are not matching between sub[ i, _ ] and tensor3[ _, _, i ]" );
            }
            let dim_number_1 = :: crate :: tensorism :: Reindexing2 :: get_input1_bound( & sub );
            if dim_number_1 != :: crate :: tensorism :: Reindexing2 :: get_input1_bound( & middle ) {
                panic! ( "Dimensions are not matching between sub[ _, j ] and middle[ _, j ]" );
            }
            if dim_number_1 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor3 ).1 {
                panic! ( "Dimensions are not matching between sub[ _, j ] and tensor3[ _, j, _ ]" );
            }
            let dim_number_2 = :: crate :: tensorism :: Reindexing1 :: get_output_bound( & sup );
            if dim_number_2 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor3 ).0 {
                panic! ( "Dimensions are not matching between k = sup[ _ ] and tensor3[ k, _, _ ]" );
            }
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 2usize] >> :: from_shape_fn(
                ( dim_number_0, dim_number_1, ),
                | ( i, j, ) | {
                    let k = crate :: tensorism :: Reindexing1 :: get_unchecked(
                        & sup,
                        crate :: tensorism :: Reindexing2 :: get_unchecked(
                            & middle,
                            crate :: tensorism :: Reindexing2 :: get_unchecked( & sub, i, j ),
                            j
                        )
                    );
                    (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor3, (k, j, i, )) })
                }
            )
        } "#
    );
}
