# GAN
GANの実験用に作成．
日付時間でファイル管理していく


[error] (run-main-2) java.lang.IllegalArgumentException: requirement failed: Row dimension mismatch!: a.rows == b.rows (256 != 1)
java.lang.IllegalArgumentException: requirement failed: Row dimension mismatch!: a.rows == b.rows (256 != 1)
        at breeze.linalg.operators.DenseMatrixOps$$anon$39.apply(DenseMatrixOps.scala:452)
        at breeze.linalg.operators.DenseMatrixOps$$anon$39.apply(DenseMatrixOps.scala:450)
        at breeze.linalg.NumericOps.$colon$plus$eq(NumericOps.scala:198)
        at breeze.linalg.NumericOps.$colon$plus$eq$(NumericOps.scala:197)
        at breeze.linalg.DenseMatrix.$colon$plus$eq(DenseMatrix.scala:52)
        at breeze.linalg.NumericOps.$plus$eq(NumericOps.scala:210)
        at breeze.linalg.NumericOps.$plus$eq$(NumericOps.scala:209)
        at breeze.linalg.DenseMatrix.$plus$eq(DenseMatrix.scala:52)
        at Affine.backward(Network.scala:69)
        at Layer.$anonfun$backward$1(Network.scala:32)
        at scala.collection.immutable.Range.foreach$mVc$sp(Range.scala:156)
        at Layer.backward(Network.scala:31)
        at gan_Network$.$anonfun$backwards$2(generalAdversalial_layer.scala:99)
        at gan_Network$.$anonfun$backwards$2$adapted(generalAdversalial_layer.scala:98)
        at scala.collection.immutable.List.foreach(List.scala:389)
        at gan_Network$.backwards(generalAdversalial_layer.scala:98)
        at GAN$.learning_G(generalAdversalial.scala:61)
        at GAN$.$anonfun$main$1(generalAdversalial.scala:35)
        at scala.collection.immutable.Range.foreach$mVc$sp(Range.scala:156)
        at GAN$.main(generalAdversalial.scala:31)
        at GAN.main(generalAdversalial.scala)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:498)
[trace] Stack trace suppr
