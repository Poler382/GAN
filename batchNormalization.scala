import breeze.linalg._
import math._

object BN{
  def sub(y:Array[DenseVector[Double]],a:Array[DenseVector[Double]] )={
    var return_d = new Array[DenseVector[Double]](a.size)
    var d = Array.ofDim[Double](a(0).size)

    for(i <- 0 until a.size){
      for(j <- 0 until a(0).size){
        d(j) = y(i)(j) - a(i)(j)
      }
      return_d(i) = DenseVector(d)
    }
    return_d
  }

  def test()={
    val bn   = new BN(2,3)
    val test = Array(
      DenseVector(1d,2d),
      DenseVector(3d,4d),
      DenseVector(5d,6d)
    )

    val test2 = Array(
      DenseVector(1d,0d),
      DenseVector(2d,1d),
      DenseVector(1d,2d)
    )

    val l = bn.forward(test)

    bn.backward(test2)
  }
}
         //D:各データの個数　データ数dxxc 
class BN(val D:Int,val n: Int){
  var xn = D
  var gamma   = DenseVector.ones[Double](D)
  var beta    = DenseVector.zeros[Double](D)
  var d_beta  = DenseVector.zeros[Double](D)
  var d_gamma = DenseVector.zeros[Double](D)
  var mu      = DenseVector.zeros[Double](D)
  var eps     = 0.00000001
  var sigma   = DenseVector.zeros[Double](D)
  var x_h     = new Array[DenseVector[Double]](D)
  var x_m     = new Array[DenseVector[Double]](D)
  var xmu     = new Array[DenseVector[Double]](D)
  var dg      = DenseVector.zeros[Double](xn)
  var db      = DenseVector.zeros[Double](xn)
  var count   = 0

  def forward(xs:Array[DenseVector[Double]])={
    var y = new Array[DenseVector[Double]](xs.size)
    
    for(i <- 0 until xs.size){
      for (j <- 0 until xn){
        mu(j)+=xs(i)(j)/xs.size
      }
    }

    x_m = new Array[DenseVector[Double]](xs.size)
    
    for(i <- 0 until xs.size){
      x_m(i)=DenseVector.zeros[Double](xn)
      for(j <- 0 until xn){
        x_m(i)(j) = xs(i)(j) - mu(j)
  
        sigma(j) += x_m(i)(j)* x_m(i)(j)/xs.size
      }
    }

    x_h=new Array[DenseVector[Double]](xs.size)
    for(i <- 0 until xs.size){
      y(i)=DenseVector.zeros[Double](xn)
      x_h(i)=DenseVector.zeros[Double](xn)
      for (j <- 0 until xn){
        x_h(i)(j)= (x_m(i)(j))/math.sqrt(sigma(j)+eps)
        y(i)(j)=gamma(j)*x_h(i)(j)+beta(j)
      }
    }
    y
  }



//たてよこの行列を潰して各データのsumに変える
  def sumMatrix(in:Array[DenseVector[Double]])={
    var m  = DenseVector.zeros[Double](D)
    for(j <- 0 until D){
      for(i <- 0 until n){
        m(j) += in(i)(j)
      }
    }
    m
  }

  def backward(d:Array[DenseVector[Double]])={
    var d_beta = sumMatrix(d)
    var dx = new Array[DenseVector[Double]](n)
  
    for(i <- 0 until n){
      dx(i) = DenseVector.zeros[Double](D)
    }

    for(j <- 0 until D){
      for(i <- 0 until n){
        d_gamma(j) += d(i)(j) * x_h(i)(j)
      }
    }

    //各次元ごとに計算
    for(j <- 0 until D ){
      var d2 = 0d
      var d1 = DenseVector.zeros[Double](n)
      for(i <- 0 until n ){
        d1(i) = gamma(j) * d(i)(j)
        d2 += x_m(i)(j) *d1(i)
      }
      
      var d3 = - d2 / (sigma(j)+eps) 
      var d4 = d3 / (2*sqrt(sigma(j)+eps))

      var d8 = 0d
      var d6 = DenseVector.zeros[Double](n)
      var d7 = DenseVector.zeros[Double](n)
     
      for(i <- 0 until n){
        var d5 = 1 / n.toDouble *d4
        d6(i) = 2 * x_m(i)(j) * d5
        d7(i) = d1(i) * 1/sqrt(sigma(j)+eps)
        d8 -= d6(i) + d7(i)
      }
      for(i <- 0 until n){
        var d9 = 1 / n.toDouble * d8
        var d10 = d6(i)+d7(i)
        dx(i)(j) = d9 + d10
      }

    }
    dx
  }
  var adam_b = new Adam_DV(xn)
  var adam_g = new Adam_DV(xn)
  def update(){
    adam_b.update(beta,d_beta,n)
    adam_g.update(gamma,d_gamma,n)
    reset()
  }
  def reset(){
    db = DenseVector.zeros[Double](xn)
    dg = DenseVector.zeros[Double](xn)
    count=0
  }

}



class Adam_DM(val rows:Int, val cols:Int) {
  val eps = 0.001
  val delta = 1e-8
  val rho1 = 0.9
  val rho2 = 0.999
  var rho1t = 1d
  var rho2t = 1d
  var s = DenseMatrix.zeros[Double](rows,cols)
  var r = DenseMatrix.zeros[Double](rows,cols)

  def update(K:DenseMatrix[Double], dK:DenseMatrix[Double],count:Int) = {
    rho1t *= rho1
    rho2t *= rho2
    val rho1tr = 1 / (1 - rho1t)
    val rho2tr = 1 / (1 - rho2t)
    for(i <- 0 until K.rows; j <- 0 until K.cols) {
      s(i,j) = rho1 * s(i,j) + (1 - rho1) * dK(i,j)
      r(i,j) = rho2 * r(i,j) + (1 - rho2) * dK(i,j) * dK(i,j)
      val d = (s(i,j) * rho1tr) / (math.sqrt(r(i,j) * rho2tr) + delta)
      K(i,j) = K(i,j) - eps/count * d
    }
  }
}
class Adam_DV(val n:Int) {
  val eps = 0.001
  val delta = 1e-8
  val rho1 = 0.9
  val rho2 = 0.999
  var rho1t = 1d
  var rho2t = 1d
  var s = DenseVector.zeros[Double](n)
  var r = DenseVector.zeros[Double](n)

  def update(K:DenseVector[Double], dK:DenseVector[Double],count:Int) = {
    rho1t *= rho1
    rho2t *= rho2
    val rho1tr = 1 / (1 - rho1t)
    val rho2tr = 1 / (1 - rho2t)
    for(i <- 0 until K.size) {
      s(i) = rho1 * s(i) + (1 - rho1) * dK(i)
      r(i) = rho2 * r(i) + (1 - rho2) * dK(i) * dK(i)
      val d = (s(i) * rho1tr) / (math.sqrt(r(i) * rho2tr) + delta)
      K(i) = K(i) - eps/count * d
    }
  }
}
