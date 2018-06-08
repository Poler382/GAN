object GAN{
  val rand=new util.Random(0)

  def main(args:Array[String]){

    val mode = args(0)
    val ln   = args(1).toInt
    val dn   = args(2).toInt
    val where = args(3)

    val mn = new mnist()

    val (dtrain, dtest) = "lab" match {
      case "home" =>{
        mn.load_mnist("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" =>{
        mn.load_mnist("/home/share/fashion-mnist")
      }
    }

    println("finish load")


    val G = gan_Network.select_G(mode)
    val D = gan_Network.select_D(mode)

    println("learning start")
    val namelist =List("lossG","lossD","real rate","fake rate") 

    for(i <- 0 until ln){
      val start = System.currentTimeMillis
      val (lossD,counter1,counter2)= learning_D(where,i,dn,G,D,dtrain)
      val lossG = learning_G(where,i,dn,G,D)
      val time =  System.currentTimeMillis - start
      
      learning.print_result2(i,time,namelist,
        List(lossG,lossD,counter1*100/(dn/2),counter2*100/(dn/2)),1)

    }
  }

  //
  def learning_G(where:String,ln:Int,dn:Int,G:List[Layer],D:List[Layer])={


    var lossG = 0d
    var ys  = List[Array[Double]]()
    
    var z = new Array[Array[Double]](100)
    for(i <- 0 until 100){// make noise
      val t = new Array[Double](100).map(_ => rand.nextGaussian * 0.1)
      z(i) = t
    }

    val y =  gan_Network.forwards(G,z)
    val y2 =  gan_Network.forwards(D,y)
    lossG = y2.map(a=> math.log((1d-a(0))+1e-8)).sum

    val d  = gan_Network.backwards((G++D).reverse,y2.map(a=>a.map(b =>(-1d/b))))
   
    gan_Network.updates(G)
    gan_Network.resets(D)
    if(ln %100 == 0){
      Image.write("GAN/train"+ln.toString+".png",Image.make_image3(y.toArray,10,10,28,28))
    }

    lossG
  }

  def learning_D(where:String,ln:Int,dn:Int,G:List[Layer],D:List[Layer],dtrain:Array[(Array[Double], Int)])={

    var lossD =0d
    var fake_counter  = 0
    var real_counter  = 0

    val xn = rand.shuffle(dtrain.toList).take(dn)
    val xs = xn.map(_._1).toArray
    val y =  gan_Network.forwards(D,xs)

    for(i <- 0 until dn){
      if(y(i)(0) > 0.5){ //本物を見つける
        real_counter += 1
        //println("real "+y(0)+" realcounter "+real_counter)
      }
    }

    lossD += y.map(a => Math.log( a(0) + 0.00000001)).sum

    val d1 = gan_Network.backwards(D.reverse,y.map(a => a.map(b => -1d/b)))

    gan_Network.updates(D)

    
    var z = new Array[Array[Double]](dn)
    for(i <- 0 until dn){
     val z1 = new Array[Double](dn).map(_ => rand.nextGaussian * 0.1)
      z(i) = z1
    }

    val yy =  gan_Network.forwards(D, gan_Network.forwards(G,z))

    for(i <- 0 until dn){
      if(yy(i)(0) < 0.5){//偽者を見破る
        fake_counter +=1
      }
    }
 
    lossD += yy.map(a => Math.log( a(0) + 0.00000001)).sum

    val d2 = gan_Network.backwards(D.reverse,yy.map(a => a.map(b => -1d/b)))

    gan_Network.updates(D)

    (lossD,real_counter.toDouble,fake_counter.toDouble)

  }
}
