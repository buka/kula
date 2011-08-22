Kula
====

Just a little exploration in GPU actors using Scala, [Akka](http://akka.io) and CUDA using the [JCuda](http://www.jcuda.org/) library.   And by the way, I'm a big fan of Olivier Chafik's [ScalaCL](http://code.google.com/p/scalacl/) project - do go and check that out.

Setup
-----

First install [CUDA](http://developer.nvidia.com/cuda-downloads) and then JCuda.  Kula has been tested on both Snow Leopard and Ubuntu 11.04 (Natty).  Everything works 'out of the box' on OSX but for Linux, especially your later distributions, you need a few tricks:

### GCC

If you have GCC 4.5 (or higher) installed, you need to install GCC 4.4 and 'downgrade' your default compiler.  If you're using Ubuntu check out this from Ben McCann (http://www.benmccann.com/dev-blog/installing-cuda-and-theano/) and use this bit:

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.5 40 --slave /usr/bin/g++ g++ /usr/bin/g++-4.5
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.4 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.4
    sudo update-alternatives --config gcc'

### Installing the Dev Driver

The HDfpga blog has a great bit of help for installing the nvidia driver for Ubuntu [here](http://hdfpga.blogspot.com/2011/05/install-cuda-40-on-ubuntu-1104.html).  Just a few notes though: first, I used the aforementioned technique with GCC and second, you might find some oddities when and if you reboot your machine. I've had to reinstall the driver a few times.

### NVCC

If you will be compiling any kernel source, make sure the path to the nvcc compiler is on your path.

### JCuda Libraries

Copy all the JCuda jars into the lib/ folder.  Depending on when you read this, you might already have a version that came down with the git tarball.  On OSX, the only snag I ran into was getting the dylibs lined up correctly, it might be easier just to leave them in lib/ as well.  Check out [this forum post](http://forum.byte-welt.de/showthread.php?t=2972) about it...  On Linux, the included config.sh script should do the trick.

Features
--------

1. Kula let's you interact with your GPU through Akka actors and futures
2. Load pre-compiled CUBIN and PTX kernels
3. Compile on-the-fly CU and inline source
4. Auto-clean or pin data on the GPU for reuse
5. Launch kernels and deal with results as a future
6. Load modules and treat functions separately

### A Sample?

JCudaVectorAdd.java -> Kula...

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

### Caveats...

There are plenty. Lots of stuff is missing and/or untested (steams, shared memory, multiple devices, etc etc etc).   
This is not anywhere even near 0.1 let alone something that any sane person would attempt to use for real, especially if their income and/or reputation might depend on it.