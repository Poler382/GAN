import math._
import breeze.linalg._

abstract class Layer {
  def forward(x:Array[Double]) : Array[Double]
  def backward(x:Array[Double]) : Array[Double]
  def update() : Unit
  def reset() : Unit
  def load(fn:String) {}
  def save(fn:String) {}
}


class Affine(val xn:Int, val yn:Int) extends Layer{
  val rand = new scala.util.Random(0)
  var W = DenseMatrix.zeros[Double](yn,xn).map(_ => rand.nextGaussian*0.01)
  for(i <- 0 until yn;j <- 0 until xn){
    W(i,j)=rand.nextGaussian*0.01
  }
  var b = DenseVector.zeros[Double](yn)
  var dW = DenseMatrix.zeros[Double](yn,xn)
  var db = DenseVector.zeros[Double](yn)
  var xs = List[Array[Double]]()
  var t=0
  def push(x:Array[Double]) = { xs ::= x; x }
  def pop() = { val x = xs.head; xs = xs.tail; x }

  def forward(x:Array[Double]) = {
    push(x)
    val xv = DenseVector(x)
    val y = W * xv + b
    y.toArray
  }

  def backward(d:Array[Double]) = {
    val x = pop()
    val dv = DenseVector(d)
    val X = DenseVector(x)
    // dW,dbを計算する ★
    dW += dv * X.t
    db += dv
    var dx = DenseVector.zeros[Double](xn)
    // dxを計算する ★
    dx = W.t * dv
    dx.toArray
  }
  var rt1=1d
  var rt2=1d
  var sW = DenseMatrix.zeros[Double](yn,xn)
  var rW = DenseMatrix.zeros[Double](yn,xn)
  var sb =  DenseVector.zeros[Double](yn)
  var rb =  DenseVector.zeros[Double](yn)

  def update() {
    // W,bを更新する ★
    val epsilon = 0.001
    val rho1=0.9
    val rho2=0.999
    val delta=0.000000001
    var d_tW =DenseMatrix.zeros[Double](yn,xn)
   
    var s_hW = DenseMatrix.zeros[Double](yn,xn)
    var r_hW = DenseMatrix.zeros[Double](yn,xn)

    var d_tb = DenseVector.zeros[Double](yn)
    var s_hb =  DenseVector.zeros[Double](yn)
    var r_hb =  DenseVector.zeros[Double](yn)

    rt1=rt1*rho1
    rt2=rt2*rho2
    t=t+1
   
    for(i <- 0 until yn){
      sb(i) = rho1*sb(i)+ (1 - rho1)*db(i)
      rb(i) = rho2*rb(i) + (1 - rho2)*db(i)*db(i)
      s_hb(i) = sb(i)/(1-rt1)
      r_hb(i) = rb(i)/(1-rt2)
      d_tb(i) = - epsilon * (s_hb(i)/(Math.sqrt(r_hb(i))+delta))
      b(i) = b(i) + d_tb(i)
      for(j <- 0 until xn){
        sW(i,j) =  rho1*sW(i,j) + (1 - rho1)*dW(i,j)
        rW(i,j) =  rho2*rW(i,j) + (1 - rho2)*dW(i,j)*dW(i,j)
        s_hW(i,j) = sW(i,j)/(1-rt1)
        r_hW(i,j) = rW(i,j)/(1-rt2)
        d_tW(i,j) = - epsilon * (s_hW(i,j) /(Math.sqrt(r_hW(i,j))+delta))
        W(i,j) = W(i,j) + d_tW(i,j)
      }
    }
       reset()
  }
  def update_sgd(){
    val lr=0.01
    W -= lr * dW
    b -= lr * db
    reset()
  }
  def reset() {
    dW = DenseMatrix.zeros[Double](yn,xn)
    db = DenseVector.zeros[Double](yn)
    xs = List[Array[Double]]()
  }

  override def save(fn:String){
    val save = new java.io.PrintWriter(fn)
    for(i<-0 until yn ; j<-0 until xn){
      save.println(W(i,j))
    }
    for(i<-0 until yn){
      save.println(b(i))
    }
    save.close
  }

  override def load(fn:String){
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    for(i<-0 until yn ; j<-0 until xn){
      W(i,j) = f(i*xn + j).toDouble
    }
    for(i<-0 until yn){
      b(i) = f(yn*xn + i).toDouble
    }
  }


}

class Tanh() extends Layer{
  var ys = List[DenseVector[Double]]()
  def tanh(x:Double) = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
  def forward(xx:Array[Double]) = {
    val x = DenseVector(xx)
    ys ::= x.map(tanh)
    ys.head.toArray
  }

  def backward(d:Array[Double]) = {
    val y = ys.head
    ys = ys.tail
    val ds =DenseVector(d)
    val r = ds *:* (1d - y*y)
    r.toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[DenseVector[Double]]()
  }

}

class Sigmoid() extends Layer{
  var ys = List[DenseVector[Double]]()
  def sigmoid(x:Double) = 1 / (1 + math.exp(-x))
  def forward(xx:Array[Double]) = {
    val x = DenseVector(xx)
    ys ::= x.map(sigmoid)
    ys.head.toArray
  }

  def backward(d:Array[Double]) = {
    val ds = DenseVector(d)

    val y = ys.head
    ys = ys.tail
    val r =ds *:* y *:* (1d - y)
  
    r.toArray
  }
  def update()={
    reset()
  }
  def reset()={
    ys = List[DenseVector[Double]]()
  }

}
