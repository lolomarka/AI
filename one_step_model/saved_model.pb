??.
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??-
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name21216*
value_dtype0
o
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name21197*
value_dtype0	
?
 rnn_mod_1/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	_?*1
shared_name" rnn_mod_1/embedding_1/embeddings
?
4rnn_mod_1/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp rnn_mod_1/embedding_1/embeddings*
_output_shapes
:	_?*
dtype0
?
rnn_mod_1/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?_*)
shared_namernn_mod_1/dense_1/kernel
?
,rnn_mod_1/dense_1/kernel/Read/ReadVariableOpReadVariableOprnn_mod_1/dense_1/kernel*
_output_shapes
:	?_*
dtype0
?
rnn_mod_1/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:_*'
shared_namernn_mod_1/dense_1/bias
}
*rnn_mod_1/dense_1/bias/Read/ReadVariableOpReadVariableOprnn_mod_1/dense_1/bias*
_output_shapes
:_*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
!rnn_mod_1/gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!rnn_mod_1/gru_1/gru_cell_1/kernel
?
5rnn_mod_1/gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOp!rnn_mod_1/gru_1/gru_cell_1/kernel* 
_output_shapes
:
??*
dtype0
?
+rnn_mod_1/gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*<
shared_name-+rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel
?
?rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp+rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
rnn_mod_1/gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!rnn_mod_1/gru_1/gru_cell_1/bias
?
3rnn_mod_1/gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOprnn_mod_1/gru_1/gru_cell_1/bias*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
'Adam/rnn_mod_1/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	_?*8
shared_name)'Adam/rnn_mod_1/embedding_1/embeddings/m
?
;Adam/rnn_mod_1/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp'Adam/rnn_mod_1/embedding_1/embeddings/m*
_output_shapes
:	_?*
dtype0
?
Adam/rnn_mod_1/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?_*0
shared_name!Adam/rnn_mod_1/dense_1/kernel/m
?
3Adam/rnn_mod_1/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_mod_1/dense_1/kernel/m*
_output_shapes
:	?_*
dtype0
?
Adam/rnn_mod_1/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:_*.
shared_nameAdam/rnn_mod_1/dense_1/bias/m
?
1Adam/rnn_mod_1/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_mod_1/dense_1/bias/m*
_output_shapes
:_*
dtype0
?
(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/m
?
<Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*C
shared_name42Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/m
?
FAdam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/m
?
:Adam/rnn_mod_1/gru_1/gru_cell_1/bias/m/Read/ReadVariableOpReadVariableOp&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/m*
_output_shapes
:	?*
dtype0
?
'Adam/rnn_mod_1/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	_?*8
shared_name)'Adam/rnn_mod_1/embedding_1/embeddings/v
?
;Adam/rnn_mod_1/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp'Adam/rnn_mod_1/embedding_1/embeddings/v*
_output_shapes
:	_?*
dtype0
?
Adam/rnn_mod_1/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?_*0
shared_name!Adam/rnn_mod_1/dense_1/kernel/v
?
3Adam/rnn_mod_1/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_mod_1/dense_1/kernel/v*
_output_shapes
:	?_*
dtype0
?
Adam/rnn_mod_1/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:_*.
shared_nameAdam/rnn_mod_1/dense_1/bias/v
?
1Adam/rnn_mod_1/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_mod_1/dense_1/bias/v*
_output_shapes
:_*
dtype0
?
(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/v
?
<Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*C
shared_name42Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/v
?
FAdam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/v
?
:Adam/rnn_mod_1/gru_1/gru_cell_1/bias/v/Read/ReadVariableOpReadVariableOp&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/v*
_output_shapes
:	?*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_1Const*
_output_shapes
:_*
dtype0*?
value?B?_"?  ??                                                                                                                                                                                                                                                                                                                                                                                        
M
Const_2Const*
_output_shapes
: *
dtype0*
valueB B[UNK]
?
Const_3Const*
_output_shapes
:^*
dtype0	*?
value?B?	^"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       
?
Const_4Const*
_output_shapes
:^*
dtype0*?
value?B?^B
B B!B"B#B$B%B&B'B(B)B*B+B,B-B.B/B0B1B2B3B4B5B6B7B8B9B:B;B<B=B>B?BABBBCBDBEBFBGBHBIBJBKBLBMBNBOBPBQBRBSBTBUBVBWBXBYBZB[B\B]B_B`BaBbBcBdBeBfBgBhBiBjBkBlBmBnBoBpBqBrBsBtBuBvBwBxByBzB{B|B}B~
?
Const_5Const*
_output_shapes
:^*
dtype0*?
value?B?^B
B B!B"B#B$B%B&B'B(B)B*B+B,B-B.B/B0B1B2B3B4B5B6B7B8B9B:B;B<B=B>B?BABBBCBDBEBFBGBHBIBJBKBLBMBNBOBPBQBRBSBTBUBVBWBXBYBZB[B\B]B_B`BaBbBcBdBeBfBgBhBiBjBkBlBmBnBoBpBqBrBsBtBuBvBwBxByBzB{B|B}B~
?
Const_6Const*
_output_shapes
:^*
dtype0	*?
value?B?	^"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_3Const_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_80901
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_5Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_80909
B
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1
?4
Const_7Const"/device:CPU:0*
_output_shapes
: *
dtype0*?3
value?3B?3 B?3
m
	model
chars_from_ids
ids_from_chars
	keras_api
generate_one_step

signatures*
?
	embedding
gru
		dense

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
9
input_vocabulary
lookup_table
	keras_api* 
9
input_vocabulary
lookup_table
	keras_api* 
* 
* 
* 
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
cell

state_spec
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$_random_generator
%__call__
*&&call_and_return_all_conditional_losses*
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
?
/iter

0beta_1

1beta_2
	2decay
3learning_rateme'mf(mg4mh5mi6mjvk'vl(vm4vn5vo6vp*
.
0
41
52
63
'4
(5*
.
0
41
52
63
'4
(5*
* 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
R
<_initializer
=_create_resource
>_initialize
?_destroy_resource* 
* 
* 
R
@_initializer
A_create_resource
B_initialize
C_destroy_resource* 
* 
oi
VARIABLE_VALUE rnn_mod_1/embedding_1/embeddings5model/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?

4kernel
5recurrent_kernel
6bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M_random_generator
N__call__
*O&call_and_return_all_conditional_losses*
* 

40
51
62*

40
51
62*
* 
?

Pstates
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
 	variables
!trainable_variables
"regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
_Y
VARIABLE_VALUErnn_mod_1/dense_1/kernel-model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUErnn_mod_1/dense_1/bias+model/dense/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
RL
VARIABLE_VALUE	Adam/iter/model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEAdam/beta_11model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEAdam/beta_21model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUE
Adam/decay0model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/learning_rate8model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!rnn_mod_1/gru_1/gru_cell_1/kernel,model/variables/1/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE+rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel,model/variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErnn_mod_1/gru_1/gru_cell_1/bias,model/variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
	2*

[0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

40
51
62*

40
51
62*
* 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
8
	atotal
	bcount
c	variables
d	keras_api*
* 
* 
* 
* 
* 
YS
VARIABLE_VALUEtotal:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEcount:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

c	variables*
??
VARIABLE_VALUE'Adam/rnn_mod_1/embedding_1/embeddings/mWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/rnn_mod_1/dense_1/kernel/mOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/rnn_mod_1/dense_1/bias/mMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/mNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/mNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/mNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/rnn_mod_1/embedding_1/embeddings/vWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/rnn_mod_1/dense_1/kernel/vOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/rnn_mod_1/dense_1/bias/vMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/vNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/vNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/vNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename4rnn_mod_1/embedding_1/embeddings/Read/ReadVariableOp,rnn_mod_1/dense_1/kernel/Read/ReadVariableOp*rnn_mod_1/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp5rnn_mod_1/gru_1/gru_cell_1/kernel/Read/ReadVariableOp?rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp3rnn_mod_1/gru_1/gru_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp;Adam/rnn_mod_1/embedding_1/embeddings/m/Read/ReadVariableOp3Adam/rnn_mod_1/dense_1/kernel/m/Read/ReadVariableOp1Adam/rnn_mod_1/dense_1/bias/m/Read/ReadVariableOp<Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOpFAdam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOp:Adam/rnn_mod_1/gru_1/gru_cell_1/bias/m/Read/ReadVariableOp;Adam/rnn_mod_1/embedding_1/embeddings/v/Read/ReadVariableOp3Adam/rnn_mod_1/dense_1/kernel/v/Read/ReadVariableOp1Adam/rnn_mod_1/dense_1/bias/v/Read/ReadVariableOp<Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOpFAdam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOp:Adam/rnn_mod_1/gru_1/gru_cell_1/bias/v/Read/ReadVariableOpConst_7*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_81016
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename rnn_mod_1/embedding_1/embeddingsrnn_mod_1/dense_1/kernelrnn_mod_1/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate!rnn_mod_1/gru_1/gru_cell_1/kernel+rnn_mod_1/gru_1/gru_cell_1/recurrent_kernelrnn_mod_1/gru_1/gru_cell_1/biastotalcount'Adam/rnn_mod_1/embedding_1/embeddings/mAdam/rnn_mod_1/dense_1/kernel/mAdam/rnn_mod_1/dense_1/bias/m(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/m2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/m&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/m'Adam/rnn_mod_1/embedding_1/embeddings/vAdam/rnn_mod_1/dense_1/kernel/vAdam/rnn_mod_1/dense_1/bias/v(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/v2Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/v&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_81101ٲ,
??
?

8__inference___backward_gpu_gru_with_fallback_76431_76566
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?V
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes
:	?e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????X
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes
:	?O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:??????????q
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*#
_output_shapes
:??
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*F
_output_shapes4
2:??????????:?: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:??????????p
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes
:	?\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:??????????l

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes
:	?g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:	?:??????????:	?: :??????????:: ::??????????:?: :???:?:: ::::::: : : *<
api_implements*(gru_c6390edb-0dd7-4c8e-8f40-2d746f6f0d3d*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_76565*
go_backwards( *

time_major( :% !

_output_shapes
:	?:2.
,
_output_shapes
:??????????:%!

_output_shapes
:	?:

_output_shapes
: :2.
,
_output_shapes
:??????????: 

_output_shapes
::

_output_shapes
: :

_output_shapes
::2.
,
_output_shapes
:??????????:)	%
#
_output_shapes
:?:


_output_shapes
: :#

_output_shapes
	:???:)%
#
_output_shapes
:?: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
%__inference_gru_1_layer_call_fn_79326

inputs
initial_state_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????2?:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_78245t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????2?r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????2?:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?	
?
while_cond_77074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_77074___redundant_placeholder03
/while_while_cond_77074___redundant_placeholder13
/while_while_cond_77074___redundant_placeholder23
/while_while_cond_77074___redundant_placeholder33
/while_while_cond_77074___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_78245

inputs
initial_state0
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpr
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputsinitial_stateIdentity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:??????????:?????????2?:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_78029n

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:?????????2?j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????2?:??????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:WS
(
_output_shapes
:??????????
'
_user_specified_nameinitial_state
?	
?
while_cond_80512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_80512___redundant_placeholder03
/while_while_cond_80512___redundant_placeholder13
/while_while_cond_80512___redundant_placeholder23
/while_while_cond_80512___redundant_placeholder33
/while_while_cond_80512___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?	
?
)__inference_rnn_mod_1_layer_call_fn_77849
input_1	
unknown:	_?
	unknown_0:
??
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?_
	unknown_4:_
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_77834s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????2
!
_user_specified_name	input_1
?	
?
while_cond_79774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_79774___redundant_placeholder03
/while_while_cond_79774___redundant_placeholder13
/while_while_cond_79774___redundant_placeholder23
/while_while_cond_79774___redundant_placeholder33
/while_while_cond_79774___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?5
?
'__inference_gpu_gru_with_fallback_79940

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_60e6c111-f3cd-44c9-9834-8f38139e676c*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
while_cond_76264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_76264___redundant_placeholder03
/while_while_cond_76264___redundant_placeholder13
/while_while_cond_76264___redundant_placeholder23
/while_while_cond_76264___redundant_placeholder33
/while_while_cond_76264___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :	?: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
??
?
%__forward_gpu_gru_with_fallback_80814

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_e3f98fee-5a82-40a1-9fda-d7595a21f134*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_80679_80815*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?=
?
__inference_standard_gru_77572

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_77483*
condR
while_cond_77482*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2?[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:?????????2?Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_cf856b78-3d50-492e-9f57-45dc20376df7*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?

8__inference___backward_gpu_gru_with_fallback_75966_76101
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?V
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes
:	?e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:??????????X
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes
:	?O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:??????????q
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*#
_output_shapes
:??
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*F
_output_shapes4
2:??????????:?: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:??????????p
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes
:	?\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:??????????l

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes
:	?g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:	?:??????????:	?: :??????????:: ::??????????:?: :???:?:: ::::::: : : *<
api_implements*(gru_0daf7c50-5f77-48c3-9282-4b3a3d203123*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_76100*
go_backwards( *

time_major( :% !

_output_shapes
:	?:2.
,
_output_shapes
:??????????:%!

_output_shapes
:	?:

_output_shapes
: :2.
,
_output_shapes
:??????????: 

_output_shapes
::

_output_shapes
: :

_output_shapes
::2.
,
_output_shapes
:??????????:)	%
#
_output_shapes
:?:


_output_shapes
: :#

_output_shapes
	:???:)%
#
_output_shapes
:?: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?5
?
'__inference_gpu_gru_with_fallback_79091

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_bf5bddb3-07c4-46f8-86df-ec1cc477e798*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?=
?
__inference_standard_gru_80233

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_80144*
condR
while_cond_80143*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2?[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:?????????2?Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_3e738f8c-f8fb-471f-b15f-5c4f335065a6*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?
#__inference_generate_one_step_76144

inputs>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	?
,rnn_mod_1_embedding_1_embedding_lookup_75726:	_?@
,rnn_mod_1_gru_1_read_readvariableop_resource:
??B
.rnn_mod_1_gru_1_read_1_readvariableop_resource:
??A
.rnn_mod_1_gru_1_read_2_readvariableop_resource:	?F
3rnn_mod_1_dense_1_tensordot_readvariableop_resource:	?_?
1rnn_mod_1_dense_1_biasadd_readvariableop_resource:_	
add_y>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value
identity

identity_1??(rnn_mod_1/dense_1/BiasAdd/ReadVariableOp?*rnn_mod_1/dense_1/Tensordot/ReadVariableOp?&rnn_mod_1/embedding_1/embedding_lookup?#rnn_mod_1/gru_1/Read/ReadVariableOp?%rnn_mod_1/gru_1/Read_1/ReadVariableOp?%rnn_mod_1/gru_1/Read_2/ReadVariableOp?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2m
UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????q
UnicodeSplit/ReshapeReshapeinputs#UnicodeSplit/Reshape/shape:output:0*
T0*
_output_shapes
:?
UnicodeSplit/UnicodeDecodeUnicodeDecodeUnicodeSplit/Reshape:output:0*)
_output_shapes
::?????????*
input_encodingUTF-8n
,UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
(UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims(UnicodeSplit/UnicodeDecode:char_values:05UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0VUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMulRUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: ?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackAUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:?
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0NUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:?
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:??????????
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R?
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShapeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	?
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicecUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0qUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: ?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRange?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:??????????
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0HUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:??????????
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncodeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:?????????*
output_encodingUTF-8?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleMUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????V
RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0	*
value	B	 R _
RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorRaggedToTensor/Const:output:0!string_lookup_2/Identity:output:0RaggedToTensor/zeros:output:0'UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS?
&rnn_mod_1/embedding_1/embedding_lookupResourceGather,rnn_mod_1_embedding_1_embedding_lookup_75726,RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*?
_class5
31loc:@rnn_mod_1/embedding_1/embedding_lookup/75726*,
_output_shapes
:??????????*
dtype0?
/rnn_mod_1/embedding_1/embedding_lookup/IdentityIdentity/rnn_mod_1/embedding_1/embedding_lookup:output:0*
T0*?
_class5
31loc:@rnn_mod_1/embedding_1/embedding_lookup/75726*,
_output_shapes
:???????????
1rnn_mod_1/embedding_1/embedding_lookup/Identity_1Identity8rnn_mod_1/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????y
rnn_mod_1/ShapeShape:rnn_mod_1/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:g
rnn_mod_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
rnn_mod_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
rnn_mod_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_mod_1/strided_sliceStridedSlicernn_mod_1/Shape:output:0&rnn_mod_1/strided_slice/stack:output:0(rnn_mod_1/strided_slice/stack_1:output:0(rnn_mod_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
rnn_mod_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
rnn_mod_1/zeros/packedPack rnn_mod_1/strided_slice:output:0!rnn_mod_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Z
rnn_mod_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rnn_mod_1/zerosFillrnn_mod_1/zeros/packed:output:0rnn_mod_1/zeros/Const:output:0*
T0*
_output_shapes
:	??
#rnn_mod_1/gru_1/Read/ReadVariableOpReadVariableOp,rnn_mod_1_gru_1_read_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
rnn_mod_1/gru_1/IdentityIdentity+rnn_mod_1/gru_1/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
%rnn_mod_1/gru_1/Read_1/ReadVariableOpReadVariableOp.rnn_mod_1_gru_1_read_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
rnn_mod_1/gru_1/Identity_1Identity-rnn_mod_1/gru_1/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
%rnn_mod_1/gru_1/Read_2/ReadVariableOpReadVariableOp.rnn_mod_1_gru_1_read_2_readvariableop_resource*
_output_shapes
:	?*
dtype0
rnn_mod_1/gru_1/Identity_2Identity-rnn_mod_1/gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
rnn_mod_1/gru_1/PartitionedCallPartitionedCall:rnn_mod_1/embedding_1/embedding_lookup/Identity_1:output:0rnn_mod_1/zeros:output:0!rnn_mod_1/gru_1/Identity:output:0#rnn_mod_1/gru_1/Identity_1:output:0#rnn_mod_1/gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:	?:??????????:	?: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_75889?
*rnn_mod_1/dense_1/Tensordot/ReadVariableOpReadVariableOp3rnn_mod_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	?_*
dtype0j
 rnn_mod_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 rnn_mod_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
!rnn_mod_1/dense_1/Tensordot/ShapeShape(rnn_mod_1/gru_1/PartitionedCall:output:1*
T0*
_output_shapes
:k
)rnn_mod_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$rnn_mod_1/dense_1/Tensordot/GatherV2GatherV2*rnn_mod_1/dense_1/Tensordot/Shape:output:0)rnn_mod_1/dense_1/Tensordot/free:output:02rnn_mod_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+rnn_mod_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&rnn_mod_1/dense_1/Tensordot/GatherV2_1GatherV2*rnn_mod_1/dense_1/Tensordot/Shape:output:0)rnn_mod_1/dense_1/Tensordot/axes:output:04rnn_mod_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!rnn_mod_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
 rnn_mod_1/dense_1/Tensordot/ProdProd-rnn_mod_1/dense_1/Tensordot/GatherV2:output:0*rnn_mod_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#rnn_mod_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
"rnn_mod_1/dense_1/Tensordot/Prod_1Prod/rnn_mod_1/dense_1/Tensordot/GatherV2_1:output:0,rnn_mod_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'rnn_mod_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"rnn_mod_1/dense_1/Tensordot/concatConcatV2)rnn_mod_1/dense_1/Tensordot/free:output:0)rnn_mod_1/dense_1/Tensordot/axes:output:00rnn_mod_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
!rnn_mod_1/dense_1/Tensordot/stackPack)rnn_mod_1/dense_1/Tensordot/Prod:output:0+rnn_mod_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
%rnn_mod_1/dense_1/Tensordot/transpose	Transpose(rnn_mod_1/gru_1/PartitionedCall:output:1+rnn_mod_1/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:???????????
#rnn_mod_1/dense_1/Tensordot/ReshapeReshape)rnn_mod_1/dense_1/Tensordot/transpose:y:0*rnn_mod_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
"rnn_mod_1/dense_1/Tensordot/MatMulMatMul,rnn_mod_1/dense_1/Tensordot/Reshape:output:02rnn_mod_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_m
#rnn_mod_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_k
)rnn_mod_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$rnn_mod_1/dense_1/Tensordot/concat_1ConcatV2-rnn_mod_1/dense_1/Tensordot/GatherV2:output:0,rnn_mod_1/dense_1/Tensordot/Const_2:output:02rnn_mod_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
rnn_mod_1/dense_1/TensordotReshape,rnn_mod_1/dense_1/Tensordot/MatMul:product:0-rnn_mod_1/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????_?
(rnn_mod_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp1rnn_mod_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0?
rnn_mod_1/dense_1/BiasAddBiasAdd$rnn_mod_1/dense_1/Tensordot:output:00rnn_mod_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????_h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSlice"rnn_mod_1/dense_1/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*

begin_mask*
end_mask*
shrink_axis_maskN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??g
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*
_output_shapes

:_I
addAddV2truediv:z:0add_y*
T0*
_output_shapes

:_e
#categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :?
categorical/MultinomialMultinomialadd:z:0,categorical/Multinomial/num_samples:output:0*
T0*
_output_shapes

:y
SqueezeSqueeze categorical/Multinomial:output:0*
T0	*
_output_shapes
:*
squeeze_dims

??????????
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handleSqueeze:output:0;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0*
_output_shapes
:x
IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0^NoOp*
T0*
_output_shapes
:q

Identity_1Identity(rnn_mod_1/gru_1/PartitionedCall:output:2^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp)^rnn_mod_1/dense_1/BiasAdd/ReadVariableOp+^rnn_mod_1/dense_1/Tensordot/ReadVariableOp'^rnn_mod_1/embedding_1/embedding_lookup$^rnn_mod_1/gru_1/Read/ReadVariableOp&^rnn_mod_1/gru_1/Read_1/ReadVariableOp&^rnn_mod_1/gru_1/Read_2/ReadVariableOp.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : : : :_: : 2T
(rnn_mod_1/dense_1/BiasAdd/ReadVariableOp(rnn_mod_1/dense_1/BiasAdd/ReadVariableOp2X
*rnn_mod_1/dense_1/Tensordot/ReadVariableOp*rnn_mod_1/dense_1/Tensordot/ReadVariableOp2P
&rnn_mod_1/embedding_1/embedding_lookup&rnn_mod_1/embedding_1/embedding_lookup2J
#rnn_mod_1/gru_1/Read/ReadVariableOp#rnn_mod_1/gru_1/Read/ReadVariableOp2N
%rnn_mod_1/gru_1/Read_1/ReadVariableOp%rnn_mod_1/gru_1/Read_1/ReadVariableOp2N
%rnn_mod_1/gru_1/Read_2/ReadVariableOp%rnn_mod_1/gru_1/Read_2/ReadVariableOp2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV2:B >

_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: : 	

_output_shapes
:_:

_output_shapes
: 
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_77788

inputs
initial_state0
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpr
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputsinitial_stateIdentity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:??????????:?????????2?:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_77572n

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:?????????2?j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????2?:??????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:WS
(
_output_shapes
:??????????
'
_user_specified_nameinitial_state
??
?

8__inference___backward_gpu_gru_with_fallback_80310_80446
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:?????????2?a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:2??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:2??????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:2??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*O
_output_shapes=
;:2??????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:?????????2?u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:?????????2?u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:?????????2?:??????????: :2??????????::??????????: ::2??????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_3e738f8c-f8fb-471f-b15f-5c4f335065a6*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_80445*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:2.
,
_output_shapes
:?????????2?:.*
(
_output_shapes
:??????????:

_output_shapes
: :2.
,
_output_shapes
:2??????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:2??????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?>
?
__inference__traced_save_81016
file_prefix?
;savev2_rnn_mod_1_embedding_1_embeddings_read_readvariableop7
3savev2_rnn_mod_1_dense_1_kernel_read_readvariableop5
1savev2_rnn_mod_1_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop@
<savev2_rnn_mod_1_gru_1_gru_cell_1_kernel_read_readvariableopJ
Fsavev2_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop>
:savev2_rnn_mod_1_gru_1_gru_cell_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopF
Bsavev2_adam_rnn_mod_1_embedding_1_embeddings_m_read_readvariableop>
:savev2_adam_rnn_mod_1_dense_1_kernel_m_read_readvariableop<
8savev2_adam_rnn_mod_1_dense_1_bias_m_read_readvariableopG
Csavev2_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_m_read_readvariableopQ
Msavev2_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableopE
Asavev2_adam_rnn_mod_1_gru_1_gru_cell_1_bias_m_read_readvariableopF
Bsavev2_adam_rnn_mod_1_embedding_1_embeddings_v_read_readvariableop>
:savev2_adam_rnn_mod_1_dense_1_kernel_v_read_readvariableop<
8savev2_adam_rnn_mod_1_dense_1_bias_v_read_readvariableopG
Csavev2_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_v_read_readvariableopQ
Msavev2_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableopE
Asavev2_adam_rnn_mod_1_gru_1_gru_cell_1_bias_v_read_readvariableop
savev2_const_7

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B5model/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB-model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB+model/dense/bias/.ATTRIBUTES/VARIABLE_VALUEB/model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB0model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/3/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_rnn_mod_1_embedding_1_embeddings_read_readvariableop3savev2_rnn_mod_1_dense_1_kernel_read_readvariableop1savev2_rnn_mod_1_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop<savev2_rnn_mod_1_gru_1_gru_cell_1_kernel_read_readvariableopFsavev2_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop:savev2_rnn_mod_1_gru_1_gru_cell_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopBsavev2_adam_rnn_mod_1_embedding_1_embeddings_m_read_readvariableop:savev2_adam_rnn_mod_1_dense_1_kernel_m_read_readvariableop8savev2_adam_rnn_mod_1_dense_1_bias_m_read_readvariableopCsavev2_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_m_read_readvariableopMsavev2_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableopAsavev2_adam_rnn_mod_1_gru_1_gru_cell_1_bias_m_read_readvariableopBsavev2_adam_rnn_mod_1_embedding_1_embeddings_v_read_readvariableop:savev2_adam_rnn_mod_1_dense_1_kernel_v_read_readvariableop8savev2_adam_rnn_mod_1_dense_1_bias_v_read_readvariableopCsavev2_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_v_read_readvariableopMsavev2_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableopAsavev2_adam_rnn_mod_1_gru_1_gru_cell_1_bias_v_read_readvariableopsavev2_const_7"/device:CPU:0*
_output_shapes
 *(
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	_?:	?_:_: : : : : :
??:
??:	?: : :	_?:	?_:_:
??:
??:	?:	_?:	?_:_:
??:
??:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	_?:%!

_output_shapes
:	?_: 

_output_shapes
:_:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&	"
 
_output_shapes
:
??:&
"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	_?:%!

_output_shapes
:	?_: 

_output_shapes
:_:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:%!

_output_shapes
:	_?:%!

_output_shapes
:	?_: 

_output_shapes
:_:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: 
?j
?
!__inference__traced_restore_81101
file_prefixD
1assignvariableop_rnn_mod_1_embedding_1_embeddings:	_?>
+assignvariableop_1_rnn_mod_1_dense_1_kernel:	?_7
)assignvariableop_2_rnn_mod_1_dense_1_bias:_&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: H
4assignvariableop_8_rnn_mod_1_gru_1_gru_cell_1_kernel:
??R
>assignvariableop_9_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel:
??F
3assignvariableop_10_rnn_mod_1_gru_1_gru_cell_1_bias:	?#
assignvariableop_11_total: #
assignvariableop_12_count: N
;assignvariableop_13_adam_rnn_mod_1_embedding_1_embeddings_m:	_?F
3assignvariableop_14_adam_rnn_mod_1_dense_1_kernel_m:	?_?
1assignvariableop_15_adam_rnn_mod_1_dense_1_bias_m:_P
<assignvariableop_16_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_m:
??Z
Fassignvariableop_17_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_m:
??M
:assignvariableop_18_adam_rnn_mod_1_gru_1_gru_cell_1_bias_m:	?N
;assignvariableop_19_adam_rnn_mod_1_embedding_1_embeddings_v:	_?F
3assignvariableop_20_adam_rnn_mod_1_dense_1_kernel_v:	?_?
1assignvariableop_21_adam_rnn_mod_1_dense_1_bias_v:_P
<assignvariableop_22_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_v:
??Z
Fassignvariableop_23_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_v:
??M
:assignvariableop_24_adam_rnn_mod_1_gru_1_gru_cell_1_bias_v:	?
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B5model/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB-model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB+model/dense/bias/.ATTRIBUTES/VARIABLE_VALUEB/model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB1model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB0model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/3/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB:model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWmodel/embedding/embeddings/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBOmodel/dense/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMmodel/dense/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/1/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/2/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNmodel/variables/3/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp1assignvariableop_rnn_mod_1_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_rnn_mod_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp)assignvariableop_2_rnn_mod_1_dense_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp4assignvariableop_8_rnn_mod_1_gru_1_gru_cell_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp>assignvariableop_9_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_rnn_mod_1_gru_1_gru_cell_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp;assignvariableop_13_adam_rnn_mod_1_embedding_1_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp3assignvariableop_14_adam_rnn_mod_1_dense_1_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_adam_rnn_mod_1_dense_1_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp<assignvariableop_16_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpFassignvariableop_17_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_adam_rnn_mod_1_gru_1_gru_cell_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp;assignvariableop_19_adam_rnn_mod_1_embedding_1_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_rnn_mod_1_dense_1_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp1assignvariableop_21_adam_rnn_mod_1_dense_1_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp<assignvariableop_22_adam_rnn_mod_1_gru_1_gru_cell_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpFassignvariableop_23_adam_rnn_mod_1_gru_1_gru_cell_1_recurrent_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp:assignvariableop_24_adam_rnn_mod_1_gru_1_gru_cell_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?,
?
while_body_77940
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_77380

inputs0
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????r
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:??????????:???????????????????:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_77164w

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*5
_output_shapes#
!:???????????????????j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
#__inference_generate_one_step_76609

inputs

states>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	?
,rnn_mod_1_embedding_1_embedding_lookup_76200:	_?@
,rnn_mod_1_gru_1_read_readvariableop_resource:
??B
.rnn_mod_1_gru_1_read_1_readvariableop_resource:
??A
.rnn_mod_1_gru_1_read_2_readvariableop_resource:	?F
3rnn_mod_1_dense_1_tensordot_readvariableop_resource:	?_?
1rnn_mod_1_dense_1_biasadd_readvariableop_resource:_	
add_y>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value
identity

identity_1??(rnn_mod_1/dense_1/BiasAdd/ReadVariableOp?*rnn_mod_1/dense_1/Tensordot/ReadVariableOp?&rnn_mod_1/embedding_1/embedding_lookup?#rnn_mod_1/gru_1/Read/ReadVariableOp?%rnn_mod_1/gru_1/Read_1/ReadVariableOp?%rnn_mod_1/gru_1/Read_2/ReadVariableOp?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2m
UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????q
UnicodeSplit/ReshapeReshapeinputs#UnicodeSplit/Reshape/shape:output:0*
T0*
_output_shapes
:?
UnicodeSplit/UnicodeDecodeUnicodeDecodeUnicodeSplit/Reshape:output:0*)
_output_shapes
::?????????*
input_encodingUTF-8n
,UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
(UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims(UnicodeSplit/UnicodeDecode:char_values:05UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0VUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMulRUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: ?
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackAUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:?
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0NUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:?
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:??????????
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R?
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShapeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	?
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicecUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0qUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: ?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRange?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:??????????
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0HUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:??????????
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncodeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:?????????*
output_encodingUTF-8?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleMUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????V
RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0	*
value	B	 R _
RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorRaggedToTensor/Const:output:0!string_lookup_2/Identity:output:0RaggedToTensor/zeros:output:0'UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS?
&rnn_mod_1/embedding_1/embedding_lookupResourceGather,rnn_mod_1_embedding_1_embedding_lookup_76200,RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*?
_class5
31loc:@rnn_mod_1/embedding_1/embedding_lookup/76200*,
_output_shapes
:??????????*
dtype0?
/rnn_mod_1/embedding_1/embedding_lookup/IdentityIdentity/rnn_mod_1/embedding_1/embedding_lookup:output:0*
T0*?
_class5
31loc:@rnn_mod_1/embedding_1/embedding_lookup/76200*,
_output_shapes
:???????????
1rnn_mod_1/embedding_1/embedding_lookup/Identity_1Identity8rnn_mod_1/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:???????????
#rnn_mod_1/gru_1/Read/ReadVariableOpReadVariableOp,rnn_mod_1_gru_1_read_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
rnn_mod_1/gru_1/IdentityIdentity+rnn_mod_1/gru_1/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
%rnn_mod_1/gru_1/Read_1/ReadVariableOpReadVariableOp.rnn_mod_1_gru_1_read_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
rnn_mod_1/gru_1/Identity_1Identity-rnn_mod_1/gru_1/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
%rnn_mod_1/gru_1/Read_2/ReadVariableOpReadVariableOp.rnn_mod_1_gru_1_read_2_readvariableop_resource*
_output_shapes
:	?*
dtype0
rnn_mod_1/gru_1/Identity_2Identity-rnn_mod_1/gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
rnn_mod_1/gru_1/PartitionedCallPartitionedCall:rnn_mod_1/embedding_1/embedding_lookup/Identity_1:output:0states!rnn_mod_1/gru_1/Identity:output:0#rnn_mod_1/gru_1/Identity_1:output:0#rnn_mod_1/gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:	?:??????????:	?: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_76354?
*rnn_mod_1/dense_1/Tensordot/ReadVariableOpReadVariableOp3rnn_mod_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	?_*
dtype0j
 rnn_mod_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 rnn_mod_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
!rnn_mod_1/dense_1/Tensordot/ShapeShape(rnn_mod_1/gru_1/PartitionedCall:output:1*
T0*
_output_shapes
:k
)rnn_mod_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$rnn_mod_1/dense_1/Tensordot/GatherV2GatherV2*rnn_mod_1/dense_1/Tensordot/Shape:output:0)rnn_mod_1/dense_1/Tensordot/free:output:02rnn_mod_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+rnn_mod_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&rnn_mod_1/dense_1/Tensordot/GatherV2_1GatherV2*rnn_mod_1/dense_1/Tensordot/Shape:output:0)rnn_mod_1/dense_1/Tensordot/axes:output:04rnn_mod_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!rnn_mod_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
 rnn_mod_1/dense_1/Tensordot/ProdProd-rnn_mod_1/dense_1/Tensordot/GatherV2:output:0*rnn_mod_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#rnn_mod_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
"rnn_mod_1/dense_1/Tensordot/Prod_1Prod/rnn_mod_1/dense_1/Tensordot/GatherV2_1:output:0,rnn_mod_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'rnn_mod_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"rnn_mod_1/dense_1/Tensordot/concatConcatV2)rnn_mod_1/dense_1/Tensordot/free:output:0)rnn_mod_1/dense_1/Tensordot/axes:output:00rnn_mod_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
!rnn_mod_1/dense_1/Tensordot/stackPack)rnn_mod_1/dense_1/Tensordot/Prod:output:0+rnn_mod_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
%rnn_mod_1/dense_1/Tensordot/transpose	Transpose(rnn_mod_1/gru_1/PartitionedCall:output:1+rnn_mod_1/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:???????????
#rnn_mod_1/dense_1/Tensordot/ReshapeReshape)rnn_mod_1/dense_1/Tensordot/transpose:y:0*rnn_mod_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
"rnn_mod_1/dense_1/Tensordot/MatMulMatMul,rnn_mod_1/dense_1/Tensordot/Reshape:output:02rnn_mod_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_m
#rnn_mod_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_k
)rnn_mod_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$rnn_mod_1/dense_1/Tensordot/concat_1ConcatV2-rnn_mod_1/dense_1/Tensordot/GatherV2:output:0,rnn_mod_1/dense_1/Tensordot/Const_2:output:02rnn_mod_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
rnn_mod_1/dense_1/TensordotReshape,rnn_mod_1/dense_1/Tensordot/MatMul:product:0-rnn_mod_1/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????_?
(rnn_mod_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp1rnn_mod_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0?
rnn_mod_1/dense_1/BiasAddBiasAdd$rnn_mod_1/dense_1/Tensordot:output:00rnn_mod_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????_h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSlice"rnn_mod_1/dense_1/BiasAdd:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*

begin_mask*
end_mask*
shrink_axis_maskN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??g
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*
_output_shapes

:_I
addAddV2truediv:z:0add_y*
T0*
_output_shapes

:_e
#categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :?
categorical/MultinomialMultinomialadd:z:0,categorical/Multinomial/num_samples:output:0*
T0*
_output_shapes

:y
SqueezeSqueeze categorical/Multinomial:output:0*
T0	*
_output_shapes
:*
squeeze_dims

??????????
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handleSqueeze:output:0;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0*
_output_shapes
:x
IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0^NoOp*
T0*
_output_shapes
:q

Identity_1Identity(rnn_mod_1/gru_1/PartitionedCall:output:2^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp)^rnn_mod_1/dense_1/BiasAdd/ReadVariableOp+^rnn_mod_1/dense_1/Tensordot/ReadVariableOp'^rnn_mod_1/embedding_1/embedding_lookup$^rnn_mod_1/gru_1/Read/ReadVariableOp&^rnn_mod_1/gru_1/Read_1/ReadVariableOp&^rnn_mod_1/gru_1/Read_2/ReadVariableOp.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+::	?: : : : : : : : :_: : 2T
(rnn_mod_1/dense_1/BiasAdd/ReadVariableOp(rnn_mod_1/dense_1/BiasAdd/ReadVariableOp2X
*rnn_mod_1/dense_1/Tensordot/ReadVariableOp*rnn_mod_1/dense_1/Tensordot/ReadVariableOp2P
&rnn_mod_1/embedding_1/embedding_lookup&rnn_mod_1/embedding_1/embedding_lookup2J
#rnn_mod_1/gru_1/Read/ReadVariableOp#rnn_mod_1/gru_1/Read/ReadVariableOp2N
%rnn_mod_1/gru_1/Read_1/ReadVariableOp%rnn_mod_1/gru_1/Read_1/ReadVariableOp2N
%rnn_mod_1/gru_1/Read_2/ReadVariableOp%rnn_mod_1/gru_1/Read_2/ReadVariableOp2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV2:B >

_output_shapes
:
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_namestates:

_output_shapes
: : 


_output_shapes
:_:

_output_shapes
: 
??
?

8__inference___backward_gpu_gru_with_fallback_77241_77377
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????n
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:???????????????????a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:???????????????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:???????????????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*X
_output_shapesF
D:???????????????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:???????????????????u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:???????????????????:??????????: :???????????????????::??????????: ::???????????????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_53b5e9ae-acce-4918-b2aa-9b3ef5b51509*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_77376*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:;7
5
_output_shapes#
!:???????????????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :;7
5
_output_shapes#
!:???????????????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::;	7
5
_output_shapes#
!:???????????????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?

8__inference___backward_gpu_gru_with_fallback_76851_76987
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????n
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:???????????????????a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:???????????????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:???????????????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*X
_output_shapesF
D:???????????????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:???????????????????u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:???????????????????:??????????: :???????????????????::??????????: ::???????????????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_7c151256-55c0-4285-a4fc-ee6a784452e7*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_76986*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:;7
5
_output_shapes#
!:???????????????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :;7
5
_output_shapes#
!:???????????????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::;	7
5
_output_shapes#
!:???????????????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78312

inputs	$
embedding_1_78286:	_?
gru_1_78298:
??
gru_1_78300:
??
gru_1_78302:	? 
dense_1_78306:	?_
dense_1_78308:_
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_78286*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_77407a
ShapeShape,embedding_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:???????????
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0zeros:output:0gru_1_78298gru_1_78300gru_1_78302*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????2?:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_78245?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_78306dense_1_78308*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77827{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_?
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
__inference_<lambda>_809018
4key_value_init21215_lookuptableimportv2_table_handle0
,key_value_init21215_lookuptableimportv2_keys	2
.key_value_init21215_lookuptableimportv2_values
identity??'key_value_init21215/LookupTableImportV2?
'key_value_init21215/LookupTableImportV2LookupTableImportV24key_value_init21215_lookuptableimportv2_table_handle,key_value_init21215_lookuptableimportv2_keys.key_value_init21215_lookuptableimportv2_values*	
Tin0	*

Tout0*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init21215/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :^:^2R
'key_value_init21215/LookupTableImportV2'key_value_init21215/LookupTableImportV2: 

_output_shapes
:^: 

_output_shapes
:^
?5
?
'__inference_gpu_gru_with_fallback_80678

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_e3f98fee-5a82-40a1-9fda-d7595a21f134*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?4
?
'__inference_gpu_gru_with_fallback_76430

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : g

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*#
_output_shapes
:?Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*A
_output_shapes/
-:??????????:?: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????h
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes
:	?*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @V
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
:	?^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????R

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes
:	?I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????:	?:
??:
??:	?*<
api_implements*(gru_c6390edb-0dd7-4c8e-8f40-2d746f6f0d3d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?=
?
__inference_standard_gru_78029

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_77940*
condR
while_cond_77939*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2?[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:?????????2?Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_53b9bab9-493f-4c52-ad81-b392ea814520*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
:
__inference__creator_80880
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name21197*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?5
?
'__inference_gpu_gru_with_fallback_78684

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_e3b57707-919c-4ee6-b575-c545765db84d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_80080
inputs_00
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????r
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:??????????:???????????????????:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_79864w

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*5
_output_shapes#
!:???????????????????j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?

8__inference___backward_gpu_gru_with_fallback_80679_80815
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:?????????2?a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:2??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:2??????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:2??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*O
_output_shapes=
;:2??????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:?????????2?u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:?????????2?u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:?????????2?:??????????: :2??????????::??????????: ::2??????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_e3f98fee-5a82-40a1-9fda-d7595a21f134*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_80814*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:2.
,
_output_shapes
:?????????2?:.*
(
_output_shapes
:??????????:

_output_shapes
: :2.
,
_output_shapes
:2??????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:2??????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_embedding_1_layer_call_and_return_conditional_losses_77407

inputs	)
embedding_lookup_77401:	_?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_77401inputs*
Tindices0	*)
_class
loc:@embedding_lookup/77401*,
_output_shapes
:?????????2?*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/77401*,
_output_shapes
:?????????2??
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????2?x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:?????????2?Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????2: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?=
?
__inference_standard_gru_76774

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_76685*
condR
while_cond_76684*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:???????????????????Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_7c151256-55c0-4285-a4fc-ee6a784452e7*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?;
?
__inference_standard_gru_75889

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	?`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	?Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	?d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	?S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitX
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	?E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	?Z
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	?I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	?U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	?Q
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	?A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	?K
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes
:	?J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	?I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	?N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	?n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Q
_output_shapes?
=: : : : :	?: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_75800*
condR
while_cond_75799*P
output_shapes?
=: : : : :	?: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??X
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes
:	?^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????P

Identity_2Identitywhile:output:4*
T0*
_output_shapes
:	?I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????:	?:
??:
??:	?*<
api_implements*(gru_0daf7c50-5f77-48c3-9282-4b3a3d203123*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?5
?
'__inference_gpu_gru_with_fallback_79563

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_d31a6acf-74d3-4d4d-a14d-2bbbabd7b67c*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?
%__forward_gpu_gru_with_fallback_77784

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_cf856b78-3d50-492e-9f57-45dc20376df7*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_77649_77785*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?=
?
__inference_standard_gru_79487

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_79398*
condR
while_cond_79397*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:???????????????????Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_d31a6acf-74d3-4d4d-a14d-2bbbabd7b67c*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
:
__inference__creator_80862
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name21216*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_80818

inputs
initial_state_00
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpr
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputsinitial_state_0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:??????????:?????????2?:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_80602n

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:?????????2?j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????2?:??????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?,
?
while_body_78926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?	
?
%__inference_gru_1_layer_call_fn_79285
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:???????????????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_76990}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?

8__inference___backward_gpu_gru_with_fallback_79564_79700
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????n
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:???????????????????a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:???????????????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:???????????????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*X
_output_shapesF
D:???????????????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:???????????????????u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:???????????????????:??????????: :???????????????????::??????????: ::???????????????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_d31a6acf-74d3-4d4d-a14d-2bbbabd7b67c*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_79699*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:;7
5
_output_shapes#
!:???????????????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :;7
5
_output_shapes#
!:???????????????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::;	7
5
_output_shapes#
!:???????????????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78402
input_1	$
embedding_1_78376:	_?
gru_1_78388:
??
gru_1_78390:
??
gru_1_78392:	? 
dense_1_78396:	?_
dense_1_78398:_
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_1_78376*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_77407a
ShapeShape,embedding_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:???????????
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0zeros:output:0gru_1_78388gru_1_78390gru_1_78392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????2?:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_78245?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_78396dense_1_78398*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77827{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_?
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????2
!
_user_specified_name	input_1
?
?
'__inference_dense_1_layer_call_fn_80827

inputs
unknown:	?_
	unknown_0:_
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77827s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?
?
__inference__initializer_808708
4key_value_init21215_lookuptableimportv2_table_handle0
,key_value_init21215_lookuptableimportv2_keys	2
.key_value_init21215_lookuptableimportv2_values
identity??'key_value_init21215/LookupTableImportV2?
'key_value_init21215/LookupTableImportV2LookupTableImportV24key_value_init21215_lookuptableimportv2_table_handle,key_value_init21215_lookuptableimportv2_keys.key_value_init21215_lookuptableimportv2_values*	
Tin0	*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init21215/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :^:^2R
'key_value_init21215/LookupTableImportV2'key_value_init21215/LookupTableImportV2: 

_output_shapes
:^: 

_output_shapes
:^
??
?
%__forward_gpu_gru_with_fallback_76986

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_7c151256-55c0-4285-a4fc-ee6a784452e7*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_76851_76987*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_77834

inputs	$
embedding_1_77408:	_?
gru_1_77789:
??
gru_1_77791:
??
gru_1_77793:	? 
dense_1_77828:	?_
dense_1_77830:_
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_77408*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_77407a
ShapeShape,embedding_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:???????????
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0zeros:output:0gru_1_77789gru_1_77791gru_1_77793*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????2?:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_77788?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_77828dense_1_77830*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77827{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_?
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
,
__inference__destroyer_80893
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_80857

inputs4
!tensordot_readvariableop_resource:	?_-
biasadd_readvariableop_resource:_
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?_*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????2??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2_r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2_c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2_z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?	
?
while_cond_78518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_78518___redundant_placeholder03
/while_while_cond_78518___redundant_placeholder13
/while_while_cond_78518___redundant_placeholder23
/while_while_cond_78518___redundant_placeholder33
/while_while_cond_78518___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?5
?
'__inference_gpu_gru_with_fallback_78105

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_53b9bab9-493f-4c52-ad81-b392ea814520*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?5
?
'__inference_gpu_gru_with_fallback_77240

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_53b5e9ae-acce-4918-b2aa-9b3ef5b51509*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?

8__inference___backward_gpu_gru_with_fallback_78106_78242
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:?????????2?a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:2??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:2??????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:2??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*O
_output_shapes=
;:2??????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:?????????2?u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:?????????2?u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:?????????2?:??????????: :2??????????::??????????: ::2??????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_53b9bab9-493f-4c52-ad81-b392ea814520*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_78241*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:2.
,
_output_shapes
:?????????2?:.*
(
_output_shapes
:??????????:

_output_shapes
: :2.
,
_output_shapes
:2??????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:2??????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?5
?
'__inference_gpu_gru_with_fallback_80309

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_3e738f8c-f8fb-471f-b15f-5c4f335065a6*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
,
__inference__destroyer_80875
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?=
?
__inference_standard_gru_78608

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_78519*
condR
while_cond_78518*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2?[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:?????????2?Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_e3b57707-919c-4ee6-b575-c545765db84d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_79703
inputs_00
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????r
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:??????????:???????????????????:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_79487w

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*5
_output_shapes#
!:???????????????????j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_76990

inputs0
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????r
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:??????????:???????????????????:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_76774w

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*5
_output_shapes#
!:???????????????????j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
while_cond_77939
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_77939___redundant_placeholder03
/while_while_cond_77939___redundant_placeholder13
/while_while_cond_77939___redundant_placeholder23
/while_while_cond_77939___redundant_placeholder33
/while_while_cond_77939___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?,
?
while_body_79398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?
?
__inference__initializer_808888
4key_value_init21196_lookuptableimportv2_table_handle0
,key_value_init21196_lookuptableimportv2_keys2
.key_value_init21196_lookuptableimportv2_values	
identity??'key_value_init21196/LookupTableImportV2?
'key_value_init21196/LookupTableImportV2LookupTableImportV24key_value_init21196_lookuptableimportv2_table_handle,key_value_init21196_lookuptableimportv2_keys.key_value_init21196_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init21196/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :^:^2R
'key_value_init21196/LookupTableImportV2'key_value_init21196/LookupTableImportV2: 

_output_shapes
:^: 

_output_shapes
:^
?
?
while_cond_75799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_75799___redundant_placeholder03
/while_while_cond_75799___redundant_placeholder13
/while_while_cond_75799___redundant_placeholder23
/while_while_cond_75799___redundant_placeholder33
/while_while_cond_75799___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :	?: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?,
?
while_body_76685
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?	
?
while_cond_80143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_80143___redundant_placeholder03
/while_while_cond_80143___redundant_placeholder13
/while_while_cond_80143___redundant_placeholder23
/while_while_cond_80143___redundant_placeholder33
/while_while_cond_80143___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78373
input_1	$
embedding_1_78347:	_?
gru_1_78359:
??
gru_1_78361:
??
gru_1_78363:	? 
dense_1_78367:	?_
dense_1_78369:_
identity??dense_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_1_78347*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_77407a
ShapeShape,embedding_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:???????????
gru_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0zeros:output:0gru_1_78359gru_1_78361gru_1_78363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????2?:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_77788?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_1_78367dense_1_78369*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_77827{
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_?
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????2
!
_user_specified_name	input_1
?>
?
%__forward_gpu_gru_with_fallback_76565

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : g

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*#
_output_shapes
:?Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*A
_output_shapes/
-:??????????:?: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????h
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes
:	?*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @V
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
:	?^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????R

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes
:	?I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????:	?:
??:
??:	?*<
api_implements*(gru_c6390edb-0dd7-4c8e-8f40-2d746f6f0d3d*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_76431_76566*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?+
?
while_body_76265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	?s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	?W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	?y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	?Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitj
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes
:	?Q
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes
:	?l
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes
:	?U
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes
:	?g
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes
:	?c
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes
:	?M

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes
:	?d
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	?P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??c
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes
:	?[
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes
:	?`
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes
:	??
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???W
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes
:	?"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=: : : : :	?: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?	
?
)__inference_rnn_mod_1_layer_call_fn_78442

inputs	
unknown:	_?
	unknown_0:
??
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?_
	unknown_4:_
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78312s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?
%__forward_gpu_gru_with_fallback_78820

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_e3b57707-919c-4ee6-b575-c545765db84d*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_78685_78821*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?

8__inference___backward_gpu_gru_with_fallback_79092_79228
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:?????????2?a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:2??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:2??????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:2??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*O
_output_shapes=
;:2??????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:?????????2?u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:?????????2?u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:?????????2?:??????????: :2??????????::??????????: ::2??????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_bf5bddb3-07c4-46f8-86df-ec1cc477e798*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_79227*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:2.
,
_output_shapes
:?????????2?:.*
(
_output_shapes
:??????????:

_output_shapes
: :2.
,
_output_shapes
:2??????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:2??????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?=
?
__inference_standard_gru_77164

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_77075*
condR
while_cond_77074*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:???????????????????Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_53b5e9ae-acce-4918-b2aa-9b3ef5b51509*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?,
?
while_body_78519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?	
?
while_cond_77482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_77482___redundant_placeholder03
/while_while_cond_77482___redundant_placeholder13
/while_while_cond_77482___redundant_placeholder23
/while_while_cond_77482___redundant_placeholder33
/while_while_cond_77482___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?	
?
while_cond_79397
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_79397___redundant_placeholder03
/while_while_cond_79397___redundant_placeholder13
/while_while_cond_79397___redundant_placeholder23
/while_while_cond_79397___redundant_placeholder33
/while_while_cond_79397___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?;
?
__inference_standard_gru_76354

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	?`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	?Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	?d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	?S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitX
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	?E
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	?Z
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	?I
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	?U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	?Q
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	?A
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	?K
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes
:	?J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Q
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	?I
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	?N
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	?n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Q
_output_shapes?
=: : : : :	?: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_76265*
condR
while_cond_76264*P
output_shapes?
=: : : : :	?: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??X
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes
:	?^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:??????????P

Identity_2Identitywhile:output:4*
T0*
_output_shapes
:	?I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????:	?:
??:
??:	?*<
api_implements*(gru_c6390edb-0dd7-4c8e-8f40-2d746f6f0d3d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?,
?
while_body_79775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?
?
+__inference_embedding_1_layer_call_fn_79263

inputs	
unknown:	_?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????2?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_77407t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????2?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????2: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?
%__forward_gpu_gru_with_fallback_79227

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_bf5bddb3-07c4-46f8-86df-ec1cc477e798*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_79092_79228*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?	
?
%__inference_gru_1_layer_call_fn_79298
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:???????????????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_77380}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?,
?
while_body_77075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
??
?

8__inference___backward_gpu_gru_with_fallback_78685_78821
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:?????????2?a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:2??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:2??????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:2??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*O
_output_shapes=
;:2??????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:?????????2?u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:?????????2?u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:?????????2?:??????????: :2??????????::??????????: ::2??????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_e3b57707-919c-4ee6-b575-c545765db84d*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_78820*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:2.
,
_output_shapes
:?????????2?:.*
(
_output_shapes
:??????????:

_output_shapes
: :2.
,
_output_shapes
:2??????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:2??????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
%__forward_gpu_gru_with_fallback_78241

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_53b9bab9-493f-4c52-ad81-b392ea814520*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_78106_78242*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?5
?
'__inference_gpu_gru_with_fallback_76850

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_7c151256-55c0-4285-a4fc-ee6a784452e7*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?
%__forward_gpu_gru_with_fallback_80076

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_60e6c111-f3cd-44c9-9834-8f38139e676c*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_79941_80077*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?>
?
%__forward_gpu_gru_with_fallback_76100

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : g

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*#
_output_shapes
:?Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*A
_output_shapes/
-:??????????:?: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????h
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes
:	?*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @V
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
:	?^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????R

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes
:	?I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????:	?:
??:
??:	?*<
api_implements*(gru_0daf7c50-5f77-48c3-9282-4b3a3d203123*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_75966_76101*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?	
?
)__inference_rnn_mod_1_layer_call_fn_78344
input_1	
unknown:	_?
	unknown_0:
??
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?_
	unknown_4:_
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78312s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????2
!
_user_specified_name	input_1
?,
?
while_body_80144
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
??
?

8__inference___backward_gpu_gru_with_fallback_79941_80077
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????n
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:???????????????????a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:???????????????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:???????????????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*X
_output_shapesF
D:???????????????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:???????????????????u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:???????????????????u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:???????????????????:??????????: :???????????????????::??????????: ::???????????????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_60e6c111-f3cd-44c9-9834-8f38139e676c*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_80076*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:;7
5
_output_shapes#
!:???????????????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :;7
5
_output_shapes#
!:???????????????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::;	7
5
_output_shapes#
!:???????????????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
while_cond_76684
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_76684___redundant_placeholder03
/while_while_cond_76684___redundant_placeholder13
/while_while_cond_76684___redundant_placeholder23
/while_while_cond_76684___redundant_placeholder33
/while_while_cond_76684___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
??
?
%__forward_gpu_gru_with_fallback_79699

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_d31a6acf-74d3-4d4d-a14d-2bbbabd7b67c*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_79564_79700*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?5
?
'__inference_gpu_gru_with_fallback_77648

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_cf856b78-3d50-492e-9f57-45dc20376df7*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_77827

inputs4
!tensordot_readvariableop_resource:	?_-
biasadd_readvariableop_resource:_
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?_*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????2??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2_r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2_c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2_z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????2?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs
?	
?
while_cond_78925
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_78925___redundant_placeholder03
/while_while_cond_78925___redundant_placeholder13
/while_while_cond_78925___redundant_placeholder23
/while_while_cond_78925___redundant_placeholder33
/while_while_cond_78925___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :??????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
??
?

8__inference___backward_gpu_gru_with_fallback_77649_77785
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4?_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:??????????e
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:?????????2?a
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:??????????O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:2??????????*
shrink_axis_mask?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:2??????????q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:???????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:2??????????a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*O
_output_shapes=
;:2??????????:??????????: :???*
rnn_modegru?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:?????????2?u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:??????????\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:??i
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:??@i
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:??@h
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:?h
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:?i
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::?
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:???
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:??@?
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:?o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
??o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
??i
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:?i
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:?j
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:??
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:??
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
???
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:?0?
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
???
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
??m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	?s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:?????????2?u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:??????????g

Identity_2Identity$gradients/split_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
??i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????:?????????2?:??????????: :2??????????::??????????: ::2??????????:??????????: :???::??????????: ::::::: : : *<
api_implements*(gru_cf856b78-3d50-492e-9f57-45dc20376df7*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_77784*
go_backwards( *

time_major( :. *
(
_output_shapes
:??????????:2.
,
_output_shapes
:?????????2?:.*
(
_output_shapes
:??????????:

_output_shapes
: :2.
,
_output_shapes
:2??????????: 

_output_shapes
::2.
,
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:2??????????:2
.
,
_output_shapes
:??????????:

_output_shapes
: :#

_output_shapes
	:???: 

_output_shapes
::.*
(
_output_shapes
:??????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_embedding_1_layer_call_and_return_conditional_losses_79272

inputs	)
embedding_lookup_79266:	_?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_79266inputs*
Tindices0	*)
_class
loc:@embedding_lookup/79266*,
_output_shapes
:?????????2?*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/79266*,
_output_shapes
:?????????2??
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????2?x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:?????????2?Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????2: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?,
?
while_body_77483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?+
?
while_body_75800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	?s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	?W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	?y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	?Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*5
_output_shapes#
!:	?:	?:	?*
	num_splitj
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes
:	?Q
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes
:	?l
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes
:	?U
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes
:	?g
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes
:	?c
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes
:	?M

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes
:	?d
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	?P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??c
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes
:	?[
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes
:	?`
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes
:	??
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???W
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes
:	?"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=: : : : :	?: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?,
?
while_body_80513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:??????????|
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:??????????W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:???????????
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:??????????Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splits
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:??????????Z
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:??????????u
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:??????????^
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:??????????p
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:??????????l
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:??????????V

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:??????????m
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:??????????d
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:??????????i
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:???O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :???`
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:??????????"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : :??????????: : :
??:?:
??:?: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?
?=
?
__inference_standard_gru_80602

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_80513*
condR
while_cond_80512*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2?[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:?????????2?Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_e3f98fee-5a82-40a1-9fda-d7595a21f134*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
__inference_<lambda>_809098
4key_value_init21196_lookuptableimportv2_table_handle0
,key_value_init21196_lookuptableimportv2_keys2
.key_value_init21196_lookuptableimportv2_values	
identity??'key_value_init21196/LookupTableImportV2?
'key_value_init21196/LookupTableImportV2LookupTableImportV24key_value_init21196_lookuptableimportv2_table_handle,key_value_init21196_lookuptableimportv2_keys.key_value_init21196_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init21196/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :^:^2R
'key_value_init21196/LookupTableImportV2'key_value_init21196/LookupTableImportV2: 

_output_shapes
:^: 

_output_shapes
:^
?4
?
'__inference_gpu_gru_with_fallback_75965

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : g

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*#
_output_shapes
:?Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes
	:???U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*A
_output_shapes/
-:??????????:?: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:??????????h
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes
:	?*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @V
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
:	?^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:??????????R

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes
:	?I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????:	?:
??:
??:	?*<
api_implements*(gru_0daf7c50-5f77-48c3-9282-4b3a3d203123*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:GC

_output_shapes
:	?
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?=
?
__inference_standard_gru_79864

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_79775*
condR
while_cond_79774*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:???????????????????Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_60e6c111-f3cd-44c9-9834-8f38139e676c*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?=
?
__inference_standard_gru_79015

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:?:?*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:2??????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:??????????i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????T
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:??????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????R
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Z
_output_shapesH
F: : : : :??????????: : :
??:?:
??:?* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_78926*
condR
while_cond_78925*Y
output_shapesH
F: : : : :??????????: : :
??:?:
??:?*
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:2??????????*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2?[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:?????????2?Y

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_bf5bddb3-07c4-46f8-86df-ec1cc477e798*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?

?
%__inference_gru_1_layer_call_fn_79312

inputs
initial_state_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *@
_output_shapes.
,:?????????2?:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_77788t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????2?r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????2?:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?<
?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_79256

inputs	5
"embedding_1_embedding_lookup_78852:	_?6
"gru_1_read_readvariableop_resource:
??8
$gru_1_read_1_readvariableop_resource:
??7
$gru_1_read_2_readvariableop_resource:	?<
)dense_1_tensordot_readvariableop_resource:	?_5
'dense_1_biasadd_readvariableop_resource:_
identity??dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?embedding_1/embedding_lookup?gru_1/Read/ReadVariableOp?gru_1/Read_1/ReadVariableOp?gru_1/Read_2/ReadVariableOp?
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_78852inputs*
Tindices0	*5
_class+
)'loc:@embedding_1/embedding_lookup/78852*,
_output_shapes
:?????????2?*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/78852*,
_output_shapes
:?????????2??
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????2?e
ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????~
gru_1/Read/ReadVariableOpReadVariableOp"gru_1_read_readvariableop_resource* 
_output_shapes
:
??*
dtype0h
gru_1/IdentityIdentity!gru_1/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
gru_1/Read_1/ReadVariableOpReadVariableOp$gru_1_read_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0l
gru_1/Identity_1Identity#gru_1/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
gru_1/Read_2/ReadVariableOpReadVariableOp$gru_1_read_2_readvariableop_resource*
_output_shapes
:	?*
dtype0k
gru_1/Identity_2Identity#gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
gru_1/PartitionedCallPartitionedCall0embedding_1/embedding_lookup/Identity_1:output:0zeros:output:0gru_1/Identity:output:0gru_1/Identity_1:output:0gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:??????????:?????????2?:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_79015?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	?_*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
dense_1/Tensordot/ShapeShapegru_1/PartitionedCall:output:1*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposegru_1/PartitionedCall:output:1!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????2??
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2_?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2_k
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2_?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^embedding_1/embedding_lookup^gru_1/Read/ReadVariableOp^gru_1/Read_1/ReadVariableOp^gru_1/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup26
gru_1/Read/ReadVariableOpgru_1/Read/ReadVariableOp2:
gru_1/Read_1/ReadVariableOpgru_1/Read_1/ReadVariableOp2:
gru_1/Read_2/ReadVariableOpgru_1/Read_2/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
)__inference_rnn_mod_1_layer_call_fn_78425

inputs	
unknown:	_?
	unknown_0:
??
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?_
	unknown_4:_
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2_*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_77834s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?
%__forward_gpu_gru_with_fallback_77376

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:???????????????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:???????????????????q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????g

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:???????????????????[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:???????????????????:??????????:
??:
??:	?*<
api_implements*(gru_53b5e9ae-acce-4918-b2aa-9b3ef5b51509*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_77241_77377*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
??
?
%__forward_gpu_gru_with_fallback_80445

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim?c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
??:
??:
??*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:?0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:?:?:?:?:?:?*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes

:??a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:??@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
??\
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:??@\
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:?\
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:?]

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:?]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:?]

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*J
_output_shapes8
6:2??????????:??????????: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*,
_output_shapes
:?????????2?q
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:??????????^

Identity_1Identitytranspose_7:y:0*
T0*,
_output_shapes
:?????????2?[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:??????????I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????2?:??????????:
??:
??:	?*<
api_implements*(gru_3e738f8c-f8fb-471f-b15f-5c4f335065a6*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_80310_80446*
go_backwards( *

time_major( :T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinit_h:HD
 
_output_shapes
:
??
 
_user_specified_namekernel:RN
 
_output_shapes
:
??
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	?

_user_specified_namebias
?
?
@__inference_gru_1_layer_call_and_return_conditional_losses_80449

inputs
initial_state_00
read_readvariableop_resource:
??2
read_1_readvariableop_resource:
??1
read_2_readvariableop_resource:	?

identity_3

identity_4??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpr
Read/ReadVariableOpReadVariableOpread_readvariableop_resource* 
_output_shapes
:
??*
dtype0\
IdentityIdentityRead/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	?*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
PartitionedCallPartitionedCallinputsinitial_state_0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:??????????:?????????2?:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_80233n

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:?????????2?j

Identity_4IdentityPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????2?:??????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:?????????2?
 
_user_specified_nameinputs:YU
(
_output_shapes
:??????????
)
_user_specified_nameinitial_state/0
?<
?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78849

inputs	5
"embedding_1_embedding_lookup_78445:	_?6
"gru_1_read_readvariableop_resource:
??8
$gru_1_read_1_readvariableop_resource:
??7
$gru_1_read_2_readvariableop_resource:	?<
)dense_1_tensordot_readvariableop_resource:	?_5
'dense_1_biasadd_readvariableop_resource:_
identity??dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?embedding_1/embedding_lookup?gru_1/Read/ReadVariableOp?gru_1/Read_1/ReadVariableOp?gru_1/Read_2/ReadVariableOp?
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_78445inputs*
Tindices0	*5
_class+
)'loc:@embedding_1/embedding_lookup/78445*,
_output_shapes
:?????????2?*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/78445*,
_output_shapes
:?????????2??
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????2?e
ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????~
gru_1/Read/ReadVariableOpReadVariableOp"gru_1_read_readvariableop_resource* 
_output_shapes
:
??*
dtype0h
gru_1/IdentityIdentity!gru_1/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
gru_1/Read_1/ReadVariableOpReadVariableOp$gru_1_read_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0l
gru_1/Identity_1Identity#gru_1/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
gru_1/Read_2/ReadVariableOpReadVariableOp$gru_1_read_2_readvariableop_resource*
_output_shapes
:	?*
dtype0k
gru_1/Identity_2Identity#gru_1/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
gru_1/PartitionedCallPartitionedCall0embedding_1/embedding_lookup/Identity_1:output:0zeros:output:0gru_1/Identity:output:0gru_1/Identity_1:output:0gru_1/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:??????????:?????????2?:??????????: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference_standard_gru_78608?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	?_*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
dense_1/Tensordot/ShapeShapegru_1/PartitionedCall:output:1*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposegru_1/PartitionedCall:output:1!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????2??
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2_?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2_k
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2_?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^embedding_1/embedding_lookup^gru_1/Read/ReadVariableOp^gru_1/Read_1/ReadVariableOp^gru_1/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup26
gru_1/Read/ReadVariableOpgru_1/Read/ReadVariableOp2:
gru_1/Read_1/ReadVariableOpgru_1/Read_1/ReadVariableOp2:
gru_1/Read_2/ReadVariableOpgru_1/Read_2/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:?z
?
	model
chars_from_ids
ids_from_chars
	keras_api
generate_one_step

signatures"
_tf_keras_model
?
	embedding
gru
		dense

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
P
input_vocabulary
lookup_table
	keras_api"
_tf_keras_layer
P
input_vocabulary
lookup_table
	keras_api"
_tf_keras_layer
"
_generic_user_object
?2?
#__inference_generate_one_step_76144
#__inference_generate_one_step_76609?
???
FullArgSpec'
args?
jself
jinputs
jstates
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
signature_map
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$_random_generator
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/iter

0beta_1

1beta_2
	2decay
3learning_rateme'mf(mg4mh5mi6mjvk'vl(vm4vn5vo6vp"
	optimizer
J
0
41
52
63
'4
(5"
trackable_list_wrapper
J
0
41
52
63
'4
(5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_rnn_mod_1_layer_call_fn_77849
)__inference_rnn_mod_1_layer_call_fn_78425
)__inference_rnn_mod_1_layer_call_fn_78442
)__inference_rnn_mod_1_layer_call_fn_78344?
???
FullArgSpecC
args;?8
jself
jinputs
jstates
jreturn_state

jtraining
varargs
 
varkw
 
defaults?

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78849
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_79256
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78373
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78402?
???
FullArgSpecC
args;?8
jself
jinputs
jstates
jreturn_state

jtraining
varargs
 
varkw
 
defaults?

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
j
<_initializer
=_create_resource
>_initialize
?_destroy_resourceR jCustom.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
@_initializer
A_create_resource
B_initialize
C_destroy_resourceR jCustom.StaticHashTable
"
_generic_user_object
3:1	_?2 rnn_mod_1/embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_embedding_1_layer_call_fn_79263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_embedding_1_layer_call_and_return_conditional_losses_79272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?

4kernel
5recurrent_kernel
6bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M_random_generator
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Pstates
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
 	variables
!trainable_variables
"regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
%__inference_gru_1_layer_call_fn_79285
%__inference_gru_1_layer_call_fn_79298
%__inference_gru_1_layer_call_fn_79312
%__inference_gru_1_layer_call_fn_79326?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_gru_1_layer_call_and_return_conditional_losses_79703
@__inference_gru_1_layer_call_and_return_conditional_losses_80080
@__inference_gru_1_layer_call_and_return_conditional_losses_80449
@__inference_gru_1_layer_call_and_return_conditional_losses_80818?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
+:)	?_2rnn_mod_1/dense_1/kernel
$:"_2rnn_mod_1/dense_1/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_1_layer_call_fn_80827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_80857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5:3
??2!rnn_mod_1/gru_1/gru_cell_1/kernel
?:=
??2+rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel
2:0	?2rnn_mod_1/gru_1/gru_cell_1/bias
 "
trackable_list_wrapper
5
0
1
	2"
trackable_list_wrapper
'
[0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"
_generic_user_object
?2?
__inference__creator_80862?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_80870?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_80875?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
"
_generic_user_object
?2?
__inference__creator_80880?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_80888?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_80893?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
8:6	_?2'Adam/rnn_mod_1/embedding_1/embeddings/m
0:.	?_2Adam/rnn_mod_1/dense_1/kernel/m
):'_2Adam/rnn_mod_1/dense_1/bias/m
::8
??2(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/m
D:B
??22Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/m
7:5	?2&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/m
8:6	_?2'Adam/rnn_mod_1/embedding_1/embeddings/v
0:.	?_2Adam/rnn_mod_1/dense_1/kernel/v
):'_2Adam/rnn_mod_1/dense_1/bias/v
::8
??2(Adam/rnn_mod_1/gru_1/gru_cell_1/kernel/v
D:B
??22Adam/rnn_mod_1/gru_1/gru_cell_1/recurrent_kernel/v
7:5	?2&Adam/rnn_mod_1/gru_1/gru_cell_1/bias/v
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_66
__inference__creator_80862?

? 
? "? 6
__inference__creator_80880?

? 
? "? 8
__inference__destroyer_80875?

? 
? "? 8
__inference__destroyer_80893?

? 
? "? ?
__inference__initializer_80870tu?

? 
? "? ?
__inference__initializer_80888vw?

? 
? "? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_80857e'(4?1
*?'
%?"
inputs?????????2?
? ")?&
?
0?????????2_
? ?
'__inference_dense_1_layer_call_fn_80827X'(4?1
*?'
%?"
inputs?????????2?
? "??????????2_?
F__inference_embedding_1_layer_call_and_return_conditional_losses_79272`/?,
%?"
 ?
inputs?????????2	
? "*?'
 ?
0?????????2?
? ?
+__inference_embedding_1_layer_call_fn_79263S/?,
%?"
 ?
inputs?????????2	
? "??????????2??
#__inference_generate_one_step_76144_q456'(rs&?#
?
?
inputs

 
? "(?%
?
0
?
1	??
#__inference_generate_one_step_76609uq456'(rs<?9
2?/
?
inputs
?
states	?
? "(?%
?
0
?
1	??
@__inference_gru_1_layer_call_and_return_conditional_losses_79703?456P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "Z?W
P?M
+?(
0/0???????????????????
?
0/1??????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_80080?456P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "Z?W
P?M
+?(
0/0???????????????????
?
0/1??????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_80449?456m?j
c?`
%?"
inputs?????????2?

 
p 
/?,
*?'
initial_state/0??????????
? "Q?N
G?D
"?
0/0?????????2?
?
0/1??????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_80818?456m?j
c?`
%?"
inputs?????????2?

 
p
/?,
*?'
initial_state/0??????????
? "Q?N
G?D
"?
0/0?????????2?
?
0/1??????????
? ?
%__inference_gru_1_layer_call_fn_79285?456P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "L?I
)?&
0???????????????????
?
1???????????
%__inference_gru_1_layer_call_fn_79298?456P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "L?I
)?&
0???????????????????
?
1???????????
%__inference_gru_1_layer_call_fn_79312?456m?j
c?`
%?"
inputs?????????2?

 
p 
/?,
*?'
initial_state/0??????????
? "C?@
 ?
0?????????2?
?
1???????????
%__inference_gru_1_layer_call_fn_79326?456m?j
c?`
%?"
inputs?????????2?

 
p
/?,
*?'
initial_state/0??????????
? "C?@
 ?
0?????????2?
?
1???????????
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78373q456'(<?9
2?/
!?
input_1?????????2	

 
p 
p 
? ")?&
?
0?????????2_
? ?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78402q456'(<?9
2?/
!?
input_1?????????2	

 
p 
p
? ")?&
?
0?????????2_
? ?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_78849p456'(;?8
1?.
 ?
inputs?????????2	

 
p 
p 
? ")?&
?
0?????????2_
? ?
D__inference_rnn_mod_1_layer_call_and_return_conditional_losses_79256p456'(;?8
1?.
 ?
inputs?????????2	

 
p 
p
? ")?&
?
0?????????2_
? ?
)__inference_rnn_mod_1_layer_call_fn_77849d456'(<?9
2?/
!?
input_1?????????2	

 
p 
p 
? "??????????2_?
)__inference_rnn_mod_1_layer_call_fn_78344d456'(<?9
2?/
!?
input_1?????????2	

 
p 
p
? "??????????2_?
)__inference_rnn_mod_1_layer_call_fn_78425c456'(;?8
1?.
 ?
inputs?????????2	

 
p 
p 
? "??????????2_?
)__inference_rnn_mod_1_layer_call_fn_78442c456'(;?8
1?.
 ?
inputs?????????2	

 
p 
p
? "??????????2_