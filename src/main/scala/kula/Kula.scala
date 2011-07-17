package kula

import jcuda._
import jcuda.driver._
import jcuda.driver.JCudaDriver._

import akka.actor. {Actor, ActorRef}
import akka.dispatch.Dispatchers
import akka.config.Supervision. {Permanent, OneForOneStrategy}

object gpu
{
  import JCudaActor._

  private[kula] var _jac: Option[ActorRef] = None

  def apply[T](fragment: =>T)(implicit receiver:Option[ActorRef] = None) {
    if (!_jac.isDefined) {
      val jac = Actor.actorOf[JCudaActor].start
      JCudaActor.Supervisor link jac
      _jac = Some(jac)
    }
    _jac.get ! HostCode(fragment _, receiver)
  }
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

  //
  def receive = {
    case Initialize => _init
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
  case class  HostCode[T](fragment: ()=>T, receiver:Option[ActorRef])

  val Supervisor = Actor.actorOf {
    new Actor {
      self.faultHandler = OneForOneStrategy(List(classOf[Throwable]), 10, 1000)
      def receive = {
        case _ =>
      }
    }
  }
}

object KulaTestApp extends App
{
  import jcuda.runtime._

  gpu {
    val pointer = new Pointer
    JCuda.cudaMalloc(pointer, 4)
    println("Pointer: "+pointer)
    JCuda.cudaFree(pointer)
  } (Some(Actor.actorOf{new Actor{def receive = {case _ => println("done")}}}.start))
}

// supervision
// multi-devices
