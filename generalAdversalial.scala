object GAN{
  val rand=new util.Random(0)
  def main(args:Array[String]){
    val mn = new mnist()
    val (dtrain, dtest) = args(3) match {
      case "home" =>{
        mn.load_mnist("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" =>{
        mn.load_mnist("/home/share/fashion-mnist")
      }
    }

    val mode = args(0)
    val ln   = args(1).toInt
    val dn   = args(2).toInt

    val G = gan_Network.select_G(mode)
    val D = gan_Network.select_D(mode)
    
    for (i <- 0 until ln){
      //D learning
      var ygList = List[Array[Double]]()
      var zList = List[Array[Double]]()

      for((x,z)<- dtrain.take(dn)){
        val z = new Array[Double](28*28).map(_ => rand.nextGaussian )
        val yg = gan_Network.forwards(G,z)
        ygList ::= yg
      }

      val yd = gan_Network.forwards(D,ygList(0))

      val dLg = cal_dLg(yd)

      //println("create: "+i+" G -> "+yg(0)+"\tz-> " +z(0))

      val d1 = gan_Network.backwards(D,dLg)
      gan_Network.backwards(G,d1)
      gan_Network.updates(G)
      gan_Network.resets(D)


      //Glearning
    

   
    }

  }

  def learning(mode:String,ln:Int,dn:Int,G:List[Layer],D:List[Layer]){




  }

  def test(mode:String,ln:Int,dn:Int){


  }


  def cal_dLg(yd:Array[Double])={
    var returndLg = new Array[Double](yd.size)
    for(i <- 0 until yd.size){
      returndLg(i) = -1/(1-yd(i))
    }
    returndLg
  }


}
