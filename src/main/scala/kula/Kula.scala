package kula

import jcuda._
import jcuda.driver._
import jcuda.driver.JCudaDriver._

import akka.actor. {Actor, ActorRef, UntypedChannel}
import akka.event.EventHandler
import akka.config.Supervision. {Permanent, OneForOneStrategy}



object gpu
{
  import GPUActor._

  def apply[T, U](fragment: =>T)(implicit carry: (T)=>U = {}) {
    (GPUActor() ? HostCode(fragment _)) flatMap (result => carry(result))
  }
}


//abstract class JCudaKernel
//{
//  private[kula] val _launcher: KernelLauncher
//
//  var gridSize: (Int, Int) = (1,1)
//  var blockSize: (Int, Int, Int) = (1, 1, 1)
//  var sharedMemSize: Int = 0
//  var stream: Option[CUstream] = None
//
//  def apply(args: java.lang.Object*) {
//    _launcher.call(args: _*)
//  }
//}


class GPUActor extends Actor
{
  import GPUActor._
  import collection.mutable.HashMap
  import akka.dispatch.Dispatchers

  private[kula] var _device: Option[CUdevice] = None
  private[kula] var _context: Option[CUcontext] = None
  private var       _kernels = HashMap.empty[String, CUmodule]

  override def preStart {EventHandler.info(this, "Starting")}

  private def _init {
    EventHandler.info(this, "Initializing")

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

  private def _loadmod(func: String, path: String, channel: UntypedChannel) {
    try {
      val module = new CUmodule
      cuModuleLoad(module, func)

      val function = new CUfunction
      cuModuleGetFunction(function, module, func)

      _kernels += (func -> module)
      channel ! function
    }
  }

  private def _launch(kern: Kernel) {
    kernel(kern.func) match {
      case Some(future) =>
        val result = future.get
        if (result.isInstanceOf[Exception])
          throw result.asInstanceOf[Exception]

        cuLaunchKernel(
          result.asInstanceOf[CUfunction],
          kern.grid._1, kern.grid._2, kern.grid._3,
          kern.blocks._1, kern.blocks._2, kern.blocks._3,
          0, null,  // shared memory & stream for now
          Pointer.to(kern._params.map(_._1):_*),
          null
        )
        cuCtxSynchronize
        self.reply(kern)

      case None => throw kernel.KernelNotFoundException(kern.func)
    }
  }

  // Use a thread-based dispatcher to guarantee everything happens on the same thread.
  self.dispatcher = Dispatchers.newPinnedDispatcher(self)

  // For now we'll be permanent... this might change if we bind to multiple devices
  self.lifeCycle = Permanent

  //override def postRestart(reason: Throwable) {
  //  _context foreach { cuCtxDestroy _ }
  //  _context = None
  //  _device = None
  //}

  //
  def receive = {
    case Initialize => _init
    case LoadModule(func, path, channel) => _loadmod(func, path, channel)
    case HostCode(fragment, receiver) => 
      receiver match {
        case Some(actor) => actor ! fragment()
        case _init       => fragment()
      }
    case LaunchKernel(kernel: Kernel) => _launch(kernel)
    case _ =>
  }
} 

object GPUActor
{
  import akka.actor.MaximumNumberOfRestartsWithinTimeRangeReached

  val Supervisor = Actor.actorOf {
    new Actor {
      override def preStart {EventHandler.info(this, "Starting")}
      self.faultHandler = OneForOneStrategy(List(classOf[Throwable]), 10, 1000)
      def receive = {
        case MaximumNumberOfRestartsWithinTimeRangeReached(_, _, _, reason) =>
          EventHandler.error(reason, this, "Cannot restart GPU actor. Creating new instance... ")
          _Actor = None
          // TODO...
        case msg =>
          EventHandler.info(this, "Unknown message at GPU supervisor: "+msg)
      }
    }
  }.start

  private var _Actor: Option[ActorRef] = Some(apply())

  private[kula] def apply(): ActorRef = {
    if (!_Actor.isDefined) {
      val actor = Actor.actorOf[GPUActor]
      Supervisor startLink actor
      actor ! Initialize
      actor
    }
    else
      _Actor.get
  }

  case object Initialize
  case class  HostCode[T](fragment: ()=>T, receiver: Option[ActorRef])
  case class  LoadModule(func: String, path: String, channel: UntypedChannel)
  case class  LaunchKernel(kernel: Kernel)
}



