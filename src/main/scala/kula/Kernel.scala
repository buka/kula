package kula

import akka.actor.Actor
import akka.dispatch.Future



sealed class Kernel(val func: String)
{
  import java.lang.Float
  import jcuda._
  import jcuda.driver._
  import jcuda.driver.JCudaDriver._

  var grid: (Int, Int, Int)   = (1, 1, 1)
  var blocks: (Int, Int, Int) = (1, 1, 1)
  var shared: Int             = 0
  var stream: CUstream        = null

  private[kula] var _params = List.empty[(CUdeviceptr, Boolean)]

  /**
   * Adds input data
   * @param data      the input data block arranged as 1 or more vectors
   * @param vectors   the number of input vectors to send to the kernel, default is 1
   * @param direct    if true, sends the input vectors directly as arrays,
   *                  if false, sends the input vectors as pointers to the arrays, default is true
   */
  def <+ (data: Array[Float], vectors: Int = 1, direct: Boolean = true): Kernel = {
    gpu {
        //
        // Allocate data arrays on the device, one for each row. 
        // The pointers to these arrays are stored in host memory
        // then copied to the device
        //
      val stride = data.length / vectors
      val floatbytes = stride * Float.SIZE
      var hdp = List.empty[(CUdeviceptr, Boolean)]

      for (vec <- 0 until vectors) {
        val ptr = new CUdeviceptr
        cuMemAlloc(ptr, floatbytes)
        val row = vec*stride
        val vector = data.slice(row, row+stride)
        cuMemcpyHtoD(ptr, Pointer.to(vector.toArray), floatbytes)

        hdp = hdp ::: ptr
      }

        //
        // Store the host device pointers in the param list for later binding
        //
      if (direct) {
        _params = hdp :: _params
      }
        //
        // Allocate a pointer array to the input vectors and store that instead
        //
      else {
        val indirect = new CUdeviceptr
        val ptrbytes = vectors * Pointer.SIZE
        cuMemAlloc(indirect, ptrbytes)
        cuMemcpyHtoD(indirect, Pointer.to(hdp.reverse.map(_._1).toArray), ptrbytes)

        _params = (indirect, false) ::: _params
      }
    }

    this
  }

  /**
   * Reserves a kernel parameter as an output vector
   * @param width Specifies the number of elements vector
   */
  def +> (width: Int): Kernel = {
    gpu {
      val output = new CUdeviceptr
      cuMemAlloc(output, width * Float.SIZE)

      _params = (output, true) :: _params
    }

    this
  }

  /**
   * Runs this kernel on the gpu
   */
  def ^ (blocks: Int = 1)(grids: Int = 1): Future[_] = {
    grid = (grids, grid._2, grid._3)
    blocks = (blocks, blocks._2, blocks._3)
    _params = _params reverse

    GPUActor() ? GPUActor.LaunchKernel(this)
  }
}

object kernel
{
  import jcuda._
  import java.util.concurrent.ConcurrentHashMap

  //
  // keyed by kernel function name, the value is the result
  // of the compilation or loading - either a CUfunction or
  // an exception
  //
  private val _functions = new ConcurrentHashMap[String, Future[_]]

  //
  // compiles a kernel from a CU file on disk in another thread
  //
  private def _compile(func: String, path: String, options: Seq[String]): Future[_] = {
    val compiler = Actor.actorOf[KernelCompiler].start ? CompileFile(func, path, options.foldLeft("")(_+" "+_))
    _functions.put(func, compiler)
    compiler
  }

  /**
   * @param func  the function from the module to bind to the kernel
   * @param src   the cu source filepath
   * @param opt   optional parameters to pass to the nvcc
   */
  def compile(path: String)(func: String)(options: Seq[String] = Seq[String]()): Future[_] = {
    _compile(func, path, options)
  }

  //def compile(func: String)(src: => String, opt: String*): JCudaKernel = {

  def apply(name: String): Option[Future[_]] = {
    _functions.get(name) match {
      case null => None
      case future => Some(future)
    }
  }

  case class CompileFile(func: String, path: String, options: String)
  case class CompilerProcessException(reason: Exception) extends Exception
  case class KernelNotFoundException(name: String) extends Exception
}


class KernelCompiler extends Actor
{
  import kernel._
  import GPUActor._
  import akka.event.EventHandler

  override def preStart {EventHandler.info(this, "Starting")}

  def receive = {
    case CompileFile(func, path, options) =>
      val cu = io.Source.fromFile(path).mkString
      val model = "-m" + System.getProperty("sun.arch.data.model");
      val cmd = "nvcc " + model + " -ptx "+ path + " -o " + path + ".ptx " + options

      try {
        Runtime.getRuntime.exec(cmd).waitFor match {
          case 0 => GPUActor() ! LoadModule(func, path+".ptx", self)
          case _ => //throw new CompilerProcessException()
        }
      }
      catch {
        case ex: InterruptedException =>
          Thread.currentThread.interrupt
          throw new CompilerProcessException(ex)
      }
  }
}