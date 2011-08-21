package kula.samples


object SimpleActors
{
  import kula._
  import akka.actor.Actor
  import akka.dispatch.Future
  import akka.event.EventHandler

  case class GenerateInput(function: Function, length: Int)
  class Type1 extends Actor 
  {
    def receive = {
      case GenerateInput(function, length) =>
        EventHandler.debug(this, "generating input vector")
        val a = new Array[Float](length)
        for (i <- 0 until length)
          a(i) = util.Random.nextFloat
        
        EventHandler.debug(this, "copying input vector to gpu")
        function <+ a
        EventHandler.debug(this, "done")
        self.channel ! "done"
    }
  }

  def run(length: Int): Future[_] = {

    val source = """extern "C"
      __global__ void avg(int n, float *avg, float *a, float *b)
      {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
          avg[i] = (a[i] + b[i])/2.0f;
        }
    }"""

    val avg = kernel.compile("SimpleActors", source)("avg") // compile kernel and get our avg function
    val actors = List(Actor.actorOf{ new Type1 }.start, Actor.actorOf{ new Type1 }.start)

    avg <+ new Array[Int](length)                           // setup input vector length
    avg +> length                                           // setup output vector
    val input = actors.map(_ ? GenerateInput(avg, length))  // generate the input vectors

    val gridSize = 512
    val blockSize = math.ceil(length.toDouble/gridSize.toDouble).toInt

    val monitor = Actor.actorOf { new Actor {
      def receive = {
        case input:Future[_] =>
          EventHandler.debug(this, "waiting for input")
          input.get
          EventHandler.debug(this, "running kernel function")
          self.channel ! (avg^ (blockSize, gridSize)).get   // complete the future with the results
      }
    }}.start

    monitor ? Future.sequence(input)                        // launch the kernel
  }

  def main(args: Array[String]){
    val n = 250000   
    EventHandler.debug(this, "running with %d elements".format(n))
    val results = run(n).mapTo[List[Array[Float]]].get      // kernel returns a list of the output vectors
    EventHandler.debug(this, "done")

    gpu.shutdown
  }
}