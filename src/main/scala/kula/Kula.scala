package kula

import jcuda._
import jcuda.runtime._
import jcuda.driver._
import jcuda.driver.JCudaDriver._

import akka.actor. {Actor, ActorRef, UntypedChannel}
import akka.dispatch.Future
import akka.event.EventHandler
import akka.config.Supervision. {Permanent, OneForOneStrategy}


object gpu
{
  import GPU._
  import java.util.concurrent.ConcurrentHashMap
  import collection.JavaConversions._

  def apply[T](fragment: =>T): Future[_] = {
    GPU() ? HostCode(fragment _)
  }

  def shutdown {
    EventHandler.info(this, "Shutting down")
    kernel.unload
    GPU.shutdown    
  }

  private[kula] def _stash(index: Int, device: _Device) { asScalaConcurrentMap(_cache) += (index -> device) }
  private[kula] def _unstash(index: Int): Option[_Device] = { asScalaConcurrentMap(_cache) remove index }
  private val _cache = new ConcurrentHashMap[Int, _Device]
  private[kula] case class _Device(device: Option[CUdevice], context: Option[CUcontext])
}

private[kula] class GPU(index: Int) extends Actor
{
  import gpu._
  import GPU._
  import akka.dispatch.Dispatchers

  override def preStart {
    EventHandler.info(this, "Starting")
    self ! Initialize
  }

  override def postStop {
    EventHandler.info(this, "Stopping")
    _context foreach { cuCtxDestroy _ }
    _context = None
    _device = None
  }

  override def preRestart(reason: Throwable) {
    EventHandler.error(this, "Restarting due to "+reason.getMessage)
    _stash(index, _Device(_device, _context))
  }

  override def postRestart(reason: Throwable) {
    _unstash(index) match {
      case Some(_Device(device, context)) =>
        _device = device
        _context = context
      case _ =>
    }
    EventHandler.info(this, "Restarted")
  }

  private[kula] var _device: Option[CUdevice] = None
  private[kula] var _context: Option[CUcontext] = None

  private def _init {
    EventHandler.info(this, "Initializing")

    // Enable exceptions and omit all subsequent error checks
    JCudaDriver.setExceptionsEnabled(true)

    // Initialize the driver and create a context for the device.
    cuInit(0)
    val device = new CUdevice
    cuDeviceGet(device, index);
    val context = new CUcontext
    cuCtxCreate(context, index, device)

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
    case HostCode(fragment) => self.channel ! fragment()
    
    case LoadModule(path) => 
      EventHandler.info(this, "Loading kernel module from "+path)
      val module = new CUmodule
      cuModuleLoad(module, path) match {
        case CUresult.CUDA_SUCCESS => self.channel ! module
        case result => throw new KernelCUDAException(result, path)
      }


    case Launch(func: Function) =>
      EventHandler.info(this, "Launching kernel function "+func.name)
      cuLaunchKernel(
        func.function,
        func.grid._1, func.grid._2, func.grid._3,
        func.blocks._1, func.blocks._2, func.blocks._3,
        0, null,  // shared memory & stream for now
        Pointer.to(func.params.map(Pointer.to(_)) :_*),
        null
      )
      cuCtxSynchronize
      EventHandler.info(this, "Kernel function "+func.name+" complete")

      val results = func.output.map { dev =>
        val ptr = new Array[Float](dev._2)
        //val buffer = java.nio.FloatBuffer.allocate(dev._2)
        cuMemcpyDtoH(Pointer.to(ptr), dev._1, dev._2 * jcuda.Sizeof.FLOAT);   
        ptr
      }

      EventHandler.info(this, "Returning results from kernel function "+func.name)
      self.channel ! results

      // free up device memory
      func free

    case _ =>
  }
} 

object GPU
{
  import akka.actor. {MaximumNumberOfRestartsWithinTimeRangeReached, PoisonPill}

  val Supervisor = Actor.actorOf {
    new Actor {
      override def preStart {EventHandler.info(this, "Starting")}
      self.faultHandler = OneForOneStrategy(List(classOf[Throwable]), 5, 1000)
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

  private var _Actor: Option[ActorRef] = None

  private[kula] def apply(implicit device: Int = 0): ActorRef = {
    if (!_Actor.isDefined) {
      val actor = Actor.actorOf{ new GPU(device) }
      Supervisor startLink actor
      _Actor = Some(actor)
      actor
    }
    else
      _Actor.get
  }

  private [kula] def shutdown {
    EventHandler.info(this, "Stopping GPU actor")
    (GPU() ? PoisonPill).get
    _Actor = None
  }

  case object Initialize
  case class  HostCode[T](fragment: ()=>T)
  case class  LoadModule(path: String)
  case class  Launch(function: Function)
}



