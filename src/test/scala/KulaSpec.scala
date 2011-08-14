import kula._
import org.specs2.mutable._
import jcuda._
import jcuda.runtime._
import jcuda.driver._
import jcuda.driver.JCudaDriver._

class JCudaExamplesSpec extends Specification
{
  import java.util.concurrent.atomic.AtomicInteger

  val counter = new AtomicInteger(0)

  "for this version of jcuda, kula" should {
  	"run simple pointer test from web site" in {

        //
  	  	// run the fragment on the gpu
        // this will return a future that we can resolve if applicable
        // as an example, here we'll use it...
        //
  	  val future = gpu {
  	    val pointer = new Pointer
  	    JCuda.cudaMalloc(pointer, 4)
  	    println("Pointer: "+pointer)
  	    JCuda.cudaFree(pointer)

        "done"
  	  }   		

      future.mapTo[String].get must be_==("done")
  	}

  	"run JCudaVectorAdd" in {
      import kernel._

      val cupath = "./src/test/scala/JCudaVectorAddKernel.cu"

      //
      // compile the kernel & obtain the function
      //
      val adder = compile("AddKernel", cupath) ("add")
    
      val elements = 100000
      val a = new Array[Float](elements)
      val b = new Array[Float](elements)

        //
        // generate some input data
        //
      for (i <- 0 until elements) {
        a(i) = i.toFloat
        b(i) = i.toFloat
      }

      //
      // add input vectors to kernel
      //
      adder <+ Array[Int](elements)
      adder <+ a
      adder <+ b
        
      //
      // specify output vector
      //
      adder +> elements

      //
      // now run the kernel with a block size of 256
      //
      // the invocation call (^) returns a future, 
      //  which we can later resolve to a list of output arrays
      // this particular usage will wait until the computation is complete...
      //
      val gridSize = 256
      val blockSize = math.ceil(elements.toDouble/gridSize.toDouble).toInt

      val results = (adder^ (blockSize, gridSize)).mapTo[List[Array[Float]]].get
      val answer = results(0)
      //
      // verify
      //
      for (i <- 0 until answer.length) {
        (math.abs(answer(i).toDouble - (i+i).toDouble)) must be_<(1e-5)
      }

      // dummy
      counter.get must be_==(0)
    }

    "run JCudaVectorAdd compiling the kernel on the fly" in {
      import kernel._
      
      val source = """extern "C"
        __global__ void add(int n, float *a, float *b, float *sum)
        {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i<n) {
            sum[i] = a[i] + b[i];
          }
      }"""

      val kern = compile("AddFromSource", source)
      val adder = kern("add")

      val elements = 100000
      val a = new Array[Float](elements)
      val b = new Array[Float](elements)

        //
        // generate some input data
        //
      for (i <- 0 until elements) {
        a(i) = i.toFloat
        b(i) = i.toFloat
      }

      //
      // add input vectors to kernel
      //
      adder <+ Array[Int](elements)
      adder <+ a
      adder <+ b
        
      //
      // specify output vector
      //
      adder +> elements

      //
      // now run the kernel with a block size of 256
      //
      // the invocation call (^) returns a future, 
      //  which we can later resolve to a list of output arrays
      // this particular usage will wait until the computation is complete...
      //
      val gridSize = 256
      val blockSize = math.ceil(elements.toDouble/gridSize.toDouble).toInt

      val results = (adder^ (blockSize, gridSize)).mapTo[List[Array[Float]]].get
      val answer = results(0)
      //
      // verify
      //
      for (i <- 0 until answer.length) {
        (math.abs(answer(i).toDouble - (i+i).toDouble)) must be_<(1e-5)
      }

      // dummy
      counter.get must be_==(0)
    }
  }
}
