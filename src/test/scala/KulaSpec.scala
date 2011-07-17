import kula._
import org.specs2.mutable._

class JCudaExamplesSpec extends Specification
{
  "for this version of jcuda, kula" should {

  	"run simple pointer test from web site" in {
  	  import jcuda._
	  import jcuda.runtime._
	  import java.util.concurrent.atomic.AtomicInteger
	  import akka.actor.Actor

	  val n = new AtomicInteger(0)

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
  	  gpu {

      }		
  	}
  }
}
