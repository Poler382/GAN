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
    for(i <- 0 until dn){
      var z = new Array[Double](100)
      z = z.map(_ * rand.nextGaussian * 0.1)

      val y =  gan_Network.forwards(G,z)
      val y2 =  gan_Network.forwards(D,y)
      lossG += -math.log(y(0)+1e-8)
      ys ::= y

      val lgd = -1d/y(0)
      
      val d  = gan_Network.backwards(D,Array(lgd))//fix
      gan_Network.backwards(G,d)


    }
    gan_Network.updates(G)
    gan_Network.resets(D)
   if(ln %100 == 0){
     Image.write("GAN/train"+ln.toString+".png",Image.make_image3(ys.toArray,10,10,28,28))
   }

    lossG
  }

  def learning_D(where:String,ln:Int,dn:Int,G:List[Layer],D:List[Layer],dtrain:Array[(Array[Double], Int)])={

    var lossD =0d
    var fake_counter  = 0
    var real_counter  = 0
     
    var z = new Array[Double](100)
    for( (xs,n) <- rand.shuffle(dtrain.toList).take(dn/2)){

      val y =  gan_Network.forwards(D,xs)
 
      
      if(y(0) > 0.5){ //本物を見つける
      
        real_counter += 1
        println("real "+y(0)+" realcounter "+real_counter)
      }

      lossD += -math.log(y(0)+1e-8)

      val dld = -1d / y(0)

      gan_Network.backwards(D,Array(dld))
      
    }
    gan_Network.updates(D)


    for(i <- 0 until dn/2){
      var z = new Array[Double](100)
      z = z.map(_ => rand.nextGaussian * 0.1)

      val y =  gan_Network.forwards(D, gan_Network.forwards(G,z))
      
     
      if(y(0) < 0.5){//偽者を見破る
        fake_counter +=1
      }
      println("fake "+y(0)+" fakecounter "+fake_counter)
      
      lossD += -math.log(1d - y(0)+1e-8)

      val dld =  -1d/y(0)
      gan_Network.backwards(D,Array(dld))//fix
    }
    gan_Network.updates(D)
    

    (lossD,real_counter.toDouble,fake_counter.toDouble)

  }
}
