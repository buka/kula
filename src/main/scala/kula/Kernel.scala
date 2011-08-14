package kula

import akka.actor.Actor
import akka.dispatch.Future
import akka.event.EventHandler
import jcuda.driver._
import jcuda.driver.JCudaDriver._
import GPU._

case class KernelCUDAException(code: Int, path: String) extends Exception
case class KernelCompilerException(code: Int) extends Exception
case class KernelNotFoundException(name: String) extends Exception


/**
 * Represents a kernel module with potentially multiple functions
 *
 * @param name    a friendly reference name
 * @param source  one of the following:
 *                  - a source file ending in ".cu" 
 *                  - a compiled file ending in ".cubin" or ".ptx"
 *                  - a source fragment
 */
sealed class Kernel(val name: String, source: String, options: Option[Seq[String]]) 
{
  import kernel._
  import collection.mutable.HashMap

  if (source.isEmpty)
    throw new IllegalArgumentException("No kernel source found.")
  

  def apply(tag: String): Function = synchronized {
    def bind = {
      val func = new CUfunction
      val module = _module.mapTo[CUmodule].get
      cuModuleGetFunction(func, module, tag)
      
      val f = new Function(tag, func)
      _functions += (tag -> f)
      f
    }

    _functions.getOrElse(tag, bind)
  }

  def unload = synchronized {
    EventHandler.info(this, "Unloading kernel "+name)
    _module.mapTo[CUmodule].foreach { module =>
      _functions.foreach(_._2.free(true))
      _functions.clear
      gpu {cuModuleUnload(module)}
    }
  }

  private val _module: Future[_] = {
    source.split('.').toList.last match {
      case "cubin" | "ptx"  => GPU() ? LoadModule(source)
      case "cu"             => _compiler ? CompileFile(source, options)
      case _                => _compiler ? CompileSource(source, options)
    }
  }
  
  private val _functions = HashMap.empty[String, Function]

  private lazy val _compiler = Actor.actorOf{new Compiler}.start



  private class Compiler extends Actor
  {
    import akka.config.Supervision.Permanent

    self.lifeCycle = Permanent

    def receive = {
      case CompileFile(path, options) =>
        EventHandler.info(this, "Compiling kernel source from "+path)
        val model = "-m" + System.getProperty("sun.arch.data.model")
        val ptxpath = path + ".ptx"
        val cmd = "nvcc " + model + " -ptx "+ path + " -o " + ptxpath + options.getOrElse(Seq("")).foldLeft("")(_+" "+_)

        // TODO: get output/err stream

        Runtime.getRuntime.exec(cmd).waitFor match {
          case 0 => 
            EventHandler.info(this, "Kernel successfully compiled to "+ptxpath)
            GPU() forward LoadModule(ptxpath)
          
          case code => throw new KernelCompilerException(code)
        }

      case CompileSource(source, options) =>
        import java.io. {File, FileWriter}
        import java.security.MessageDigest
        import org.apache.commons.codec.binary.Base64

        val digest = Base64.encodeBase64URLSafeString(MessageDigest.getInstance("SHA").digest(source getBytes))
        val tmp = new File(System.getProperty("java.io.tmpdir") +"/kula"+ digest +".cu")
        if (tmp.exists) {
          EventHandler.info(this, "Using cached kernel source from "+tmp.getAbsolutePath)
        }
        else {
          EventHandler.info(this, "Caching kernel source to "+tmp.getAbsolutePath)
          val writer = new FileWriter(tmp)
          writer.write(source)
          writer.close
        }        
        self forward CompileFile(tmp.getAbsolutePath, options)
    }
  }
}

object kernel
{
  import jcuda._
  import java.util.concurrent.ConcurrentHashMap

  private val _kernels = new ConcurrentHashMap[String, Kernel]

  /**
   * Create or return the named kernel
   */
  def apply(tag: String, source: String = "", options: Option[Seq[String]] = None): Kernel = {
    val kern = new Kernel(tag, source, options)
    val exists = _kernels.putIfAbsent(tag, kern)
    if (exists == null)
      kern
    else
      exists
  }  

  def unload(tag: String) {
    EventHandler.info(this, "Unloading all kernels")

    val kern = _kernels.get(tag)
    if (kern != null)
      kern.unload
  }

  def unload {
    import collection.JavaConversions._

    asScalaConcurrentMap(_kernels).foreach(_._2.unload)
    _kernels.clear
  }

  /**
   * @param tag     the friendly name of the kernel
   * @param source  the kernel source, being one of:
   *                  - a file ending in ".cu" ".cubin" or ".ptx"
   *                  - inlined CUDA code
   * @param options optional parameters to pass to the nvcc
   */
  def compile(tag: String, source: String, options: Option[Seq[String]] = None): Kernel = {
    apply(tag, source, options)
  }


  case class CompileFile(path: String, options: Option[Seq[String]])
  case class CompileSource(path: String, options: Option[Seq[String]])
}


sealed class Function(val name: String, val function: CUfunction)
{
  import jcuda.{ Pointer, Sizeof }

  var grid:   (Int, Int, Int) = (1, 1, 1)
  var blocks: (Int, Int, Int) = (1, 1, 1)
  var shared: Int             = 0
  var stream: CUstream        = null

  def params: List[CUdeviceptr] = _params.map(_._1).reverse
  def output: List[(CUdeviceptr, Int)] = _params.filter(_._2 > 0).map(p => (p._1, p._2))
  private var _params = List.empty[(CUdeviceptr, Int, Boolean)]

  /**
   * Adds input data
   * This version will ensure the data is freed on the device after kernel is run [once]
   * @param data      the input data block arranged as 1 or more vectors
   * @param vectors   the number of input vectors to send to the kernel, default is 1
   */
  def <+ (data: Array[Float], vectors: Int = 1): Slot = _input(data, vectors, false)

  /**
   * Binds input data already on the device
   * This version will ensure the data is freed on the device after kernel is run [once]
   * @param slot      the reference to the device data from another function
   */
  def <+ (slot: Slot): Slot = _input(slot, false)

  /**
   * Adds input data
   * This version keeps the data on the device for subsequent use
   * @param data      the input data block arranged as 1 or more vectors
   * @param vectors   the number of input vectors to send to the kernel, default is 1
   */
  def <++ (data: Array[Float], vectors: Int = 1): Slot = _input(data, vectors, true)

  /**
   * Binds input data already on the device
   * This version keeps the data on the device for subsequent use
   * @param slot      the reference to the device data from another function
   */
  def <++ (slot: Slot): Slot = _input(slot, true)

  private def _input(slot: Slot, keep: Boolean): Slot = {
    slot match {
      case _Slot(_, devptrs) =>
        _params = devptrs.mapTo[List[(CUdeviceptr, Int)]].get.map(dp => (dp._1, dp._2, keep)) ::: _params
      case _ =>
    }
    slot
  }

  private def _input(data: Array[Float], vectors: Int, keep: Boolean): Slot = {
    _Slot(this, 
          gpu {
              //
              // Allocate data arrays on the device, one for each row. 
              // The pointers to these arrays are stored in host memory
              // then copied to the device
              //
            val stride = data.length / vectors
            val bytes = stride * Sizeof.FLOAT

            var devptrs = List.empty[(CUdeviceptr, Int)]

            for (vec <- 0 until vectors) {
              val ptr = new CUdeviceptr
              cuMemAlloc(ptr, bytes)

              val row = vec*stride
              val vector = data.slice(row, row+stride)
              cuMemcpyHtoD(ptr, Pointer.to(vector), bytes)

              devptrs = (ptr, 0) :: devptrs
              _params = (ptr, 0, keep) :: _params
            }

            devptrs
          })
  }

  /**
   * Adds input data
   * This version will ensure the data is freed on the device after kernel is run [once]
   * @param data      the input data block arranged as 1 or more vectors
   * @param vectors   the number of input vectors to send to the kernel, default is 1
   * @param direct    if true, sends the input vectors directly as arrays,
   *                  if false, sends the input vectors as pointers to the arrays, default is true
   */
  def <+ (data: Array[Int]) = {
    gpu {
        //
        // Allocate data arrays on the device, one for each row. 
        // The pointers to these arrays are stored in host memory
        // then copied to the device
        //
      val bytes = data.length * Sizeof.INT
      val ptr = new CUdeviceptr
      cuMemAlloc(ptr, bytes)
      cuMemcpyHtoD(ptr, Pointer.to(data), bytes)

      _params = (ptr, 0, true) :: _params
    }
  }

  /**
   * Reserves a kernel parameter as an output vector
   * This version will ensure the data is freed on the device after kernel is run [once]
   * @param width Specifies the number of elements vector
   */
  def +> (width: Int): Slot = _output(width, false)
  /**
   * Reserves a kernel parameter as an output vector
   * This version keeps the data on the device for subsequent use
   * @param width Specifies the number of elements vector
   */
  def ++> (width: Int): Slot = _output(width, true)

  private def _output(width: Int, keep: Boolean): Slot = {
    _Slot(this,
          gpu {
            val ptr = new CUdeviceptr
            cuMemAlloc(ptr, width * Sizeof.FLOAT)
            _params = (ptr, width, keep) :: _params

            List((ptr, width))
          })
  }

  /**
   * Runs this kernel on the gpu
   */
  def ^ (b: Int = 1, g: Int = 1): Future[_] = {
    grid = (g, grid._2, grid._3)
    blocks = (b, blocks._2, blocks._3)

    GPU() ? GPU.Launch(this)
  }

  /**
   * Clears device memory 
   * @param force if true, frees even pointers marked for keeping
   */
  def free(implicit force: Boolean = false) {
    _params.partition(_._3) match {
      case (_, List())  =>
      case (keep, free) =>
        _params = keep
        gpu {free foreach {p => cuMemFree(p._1)}}
    }
  }

  private case class _Slot(function: Function, devptrs: Future[_]) extends Slot
}

sealed trait Slot
