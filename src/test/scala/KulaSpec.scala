import kula._
import org.specs2.mutable._
import jcuda._
import jcuda.runtime._
import jcuda.driver._
import jcuda.driver.JCudaDriver._

class JCudaExamplesSpec extends Specification
{
  "for this version of jcuda, kula" should {
  	"run simple pointer test from web site" in {
	  import java.util.concurrent.atomic.AtomicInteger
	  import akka.actor.Actor

	  val n = new AtomicInteger(0)

      //
	  	// send this fragment of over to the gpu
      //
	  gpu {
	    val pointer = new Pointer
	    JCuda.cudaMalloc(pointer, 4)
	    println("Pointer: "+pointer)
	    JCuda.cudaFree(pointer)
	  } (Some(Actor.actorOf{new Actor{def receive = {case _ => n.incrementAndGet}}}.start))  		

	  Thread.sleep(1000)
	  n.get must be_==(1)
  	}

  	"run JCudaVectorAdd" in {
      import kernel._

      //
      // this version of compile loads the cu source file
      // we have two options here, either cache the future this function just returned,
      // or resolve the kernel later on
      //
      val kernel = compile ("JCudaVectorAddKernel.cu") ("add")

      val elements = 100000
      val a = new Array[Float](elements)
      val b = new Array[Float](elements)

        //
        // generate some input data
        //
      for (i <- 0 to elements) {
        a(i) = i.toFloat
        b(i) = i.toFloat
      }

      //
      // add input vectors to kernel
      //
      kernel <+ a
      kernel <+ b
        
      //
      // specify output vector
      //
      kernel +> elements

      //
      // now run the kernel ...
      //
      kernel^

      //
      // NOTE: we could also write the above as:
      //  (((kernel <+ a) <+ b) +> elements)^
      //


  	}
  }
}
