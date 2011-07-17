package kula

import jcuda._
import jcuda.driver._
import jcuda.driver.JCudaDriver._
import jcuda.utils.KernelLauncher

import akka.actor. {Actor, ActorRef}
import akka.dispatch.Dispatchers
import akka.config.Supervision. {Permanent, OneForOneStrategy}


class KernelAlreadyExistsException extends Exception

object gpu
{
  import JCudaActor._

  private[kula] var _jac: Option[ActorRef] = None

  def apply[T](fragment: =>T)(implicit receiver: Option[ActorRef] = None) {
    if (!_jac.isDefined) {
      val jac = Actor.actorOf[JCudaActor].start
      JCudaActor.Supervisor link jac
      _jac = Some(jac)
    }
    _jac.get ! HostCode(fragment _, receiver)
  }
}

object kernel
{
  import java.util.concurrent.ConcurrentHashMap

  private[kula] val _launchers = new ConcurrentHashMap[String, KernelLauncher]

  private[kula] def _insert(func: String, launcher: KernelLauncher) {
    if (_launchers.putIfAbsent(func, launcher) != null)
      throw new KernelAlreadyExistsException
  }

  object load
  {
    /**
     * @param func  the function from the module to bind to the kernel
     * @param src   the cubin filepath
     */
    def apply(func: String)(src: String) {
      _insert(func, KernelLauncher.load(src, func))
    }
  }

  object compile
  {
    /**
     * @param func  the function from the module to bind to the kernel
     * @param src   either the cu source filepath or a function returning the cu source string itself
     * @param opt   optional parameters to pass to the nvcc
     */
    def apply(func: String)(src: Either[String, () => String], opt: String*) {
      _insert (func, src match {
        case Left(path)   => KernelLauncher.create(path, func, opt: _*)
        case Right(code)  => KernelLauncher.compile(code(), func, opt: _*)
       })
    }
  }
}

trait JCudaKernel
{
  private[kula] val _launcher: KernelLauncher

  var gridSize: (Int, Int) = (1,1)
  var blockSize: (Int, Int, Int) = (1, 16, 16)
  var sharedMemSize: Int = 0
  var stream: Option[CUstream] = None
}

class JCudaActor extends Actor
{
  import JCudaActor._

  private[kula] var _device: Option[CUdevice] = None
  private[kula] var _context: Option[CUcontext] = None

  private[kula] def _init {
    // Enable exceptions and omit all subsequent error checks
    JCudaDriver.setExceptionsEnabled(true)

    // Initialize the driver and create a context for the first device.
    cuInit(0)
    val device = new CUdevice
    cuDeviceGet(device, 0);
    val context = new CUcontext
    cuCtxCreate(context, 0, device)

    _device = Some(device)
    _context = Some(context)
  }

  // Use a thread-based dispatcher to guarantee everything happens on the same thread.
  self.dispatcher = Dispatchers.newPinnedDispatcher(self)

  // For now we'll be permanent... this might change if we bind to multiple devices
  self.lifeCycle = Permanent

  override def postRestart(reason: Throwable) {
    _context foreach { cuCtxDestroy _ }
    _context = None
    _device = None
  }

  //
  def receive = {
    case Initialize                   => _init
    case HostCode(fragment, receiver) => 
      receiver match {
        case Some(actor) => actor ! fragment()
        case _init       => fragment()
      }
    case _ =>
  }
} 

object JCudaActor
{
  case object Initialize
  case class  HostCode[T](fragment: ()=>T, receiver: Option[ActorRef])

  val Supervisor = Actor.actorOf {
    new Actor {
      self.faultHandler = OneForOneStrategy(List(classOf[Throwable]), 10, 1000)
      def receive = {
        case _ =>
      }
    }
  }
}



