object gan_Network{

  def select_G(mode:String)={
    val g = mode match {
      case "0" =>{
        val a = new Affine (100,256)
        val b = new BatchNormalization2(256)
        val c = new ReLU()
        val d = new Affine(256,512)
        val e = new BatchNormalization2(512)
        val g = new ReLU()
        val h = new Affine(512,1024)
        val i = new BatchNormalization2(1024)
        val j = new ReLU()
        val k = new Affine (1024,784)
        val l = new Tanh()
        List(a,b,c,d,e,g,h,i,j,k,l)
      }
      case "00" =>{
        val a = new Affine (100,256)
        val b = new ReLU()
        val c = new Affine(256,512)
        val d = new ReLU()
        val e = new Affine(512,1024)
        val f = new ReLU()
        val g = new Affine (1024,784)
        val h = new Tanh()
        List(a,b,c,d,e,f,g,h)
      }

      case "test" =>{
        val a = new Affine(1,1)
        val b = new Tanh()
        List(a,b)
      }
    }

    g
  }


  def select_D(mode:String)={
    val d = mode match {
      case "0" => {
        val a = new Affine(784,1024)
        val b = new ReLU()
        val c = new Affine(1024,512)
        val d = new ReLU()
        val e = new Affine(512,1)
        val f = new Sigmoid()
        List(a,b,c,d,e,f)
      }
      case "test"=>{
        val a= new Affine(1,1)
        val b = new Sigmoid()
        List(a,b)
      }
    }
    d
  }


  def forwards(layers:List[Layer],x:Array[Double])={
    var temp = x
    for(lay <- layers){
      temp =lay.forward(temp)

    }
    temp
  }

  def backwards(layers:List[Layer],x:Array[Double])={
    var d = x
    for(lay <- layers.reverse){d = lay.backward(d)}
    d
  }

  def updates(layers:List[Layer])={
    for(lay <- layers){lay.update()}
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){lay.reset()}
  }

}
