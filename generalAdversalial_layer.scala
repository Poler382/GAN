object gan_Network{

  def select_G(mode:String)={
    val g = mode match {
      case "0" =>{
        val a = new Affine (100,256)
        val b = new BatchNormalization(1,256)
        val c = new ReLU()
        val d = new Affine(256,512)
        val e = new BatchNormalization(1,256)
        val g = new ReLU()
        val h = new Afiine(512,1024)
        val i = new BatchNormalization(1,256)
        val j = new ReLU()
        val k = new Affine (1024,784)
        val l = new Tanh()
        List(a,b)
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
        val a = new Affine (28*28,28*28)
        val b = new Sigmoid()
        List(a,b)
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
