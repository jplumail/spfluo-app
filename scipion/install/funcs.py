# **************************************************************************
# *
# * Authors: J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import platform
import sys
import time
from glob import glob
from os.path import join, exists, islink, abspath
from subprocess import STDOUT, call

from pyworkflow import Config
import pwem

try:
    unicode = unicode
except NameError:  # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)

# Then we get some OS vars
MACOSX = (platform.system() == 'Darwin')
WINDOWS = (platform.system() == 'Windows')
LINUX = (platform.system() == 'Linux')
VOID_TGZ = "void.tgz"


def ansi(n):
    """Return function that escapes text with ANSI color n."""
    return lambda txt: '\x1b[%dm%s\x1b[0m' % (n, txt)


black, red, green, yellow, blue, magenta, cyan, white = map(ansi, range(30, 38))


# We don't take them from pyworkflow.utils because this has to run
# with all python versions (and so it is simplified).


def progInPath(prog):
    """ Is program prog in PATH? """
    for base in os.environ.get('PATH', '').split(os.pathsep):
        if exists('%s/%s' % (base, prog)):
            return True
    return False


def checkLib(lib, target=None):
    """ See if we have library lib """
    try:
        ret = call(['pkg-config', '--cflags', '--libs', lib],
                   stdout=open(os.devnull, 'w'), stderr=STDOUT)
        if ret != 0:
            raise OSError
    except OSError:
        try:
            ret = call(['%s-config' % lib, '--cflags'])
            if ret != 0:
                raise OSError
        except OSError:
            print("""
  ************************************************************************
    Warning: %s not found. Please consider installing it first.
  ************************************************************************

Continue anyway? (y/n)""" % lib)
            if input().upper() != 'Y':
                sys.exit(2)
    # TODO: maybe write the result of the check in
    # software/log/lib_...log so we don't check again if we already said "no"


class Command:
    def __init__(self, env, cmd, targets=None, **kwargs):
        self._env = env
        self._cmd = cmd

        if targets is None:
            self._targets = []
        elif isinstance(targets, basestring):
            self._targets = [targets]
        else:
            self._targets = targets

        self._cwd = kwargs.get('cwd', None)
        self._out = kwargs.get('out', None)
        self._always = kwargs.get('always', False)
        self._environ = kwargs.get('environ', None)

    def _existsAll(self):
        """ Return True if all targets exist. """
        for t in self._targets:
            if not glob(t):
                return False
        return True

    def execute(self):
        if not self._always and self._targets and self._existsAll():
            print("  Skipping command: %s" % cyan(self._cmd))
            print("  All targets exist.")
        else:
            cwd = os.getcwd()
            if self._cwd is not None:
                if not self._env.showOnly:
                    os.chdir(self._cwd)
                print(cyan("cd %s" % self._cwd))

            # Actually allow self._cmd to be a list or a
            # '\n'-separated list of commands, and run them all.
            if isinstance(self._cmd, basestring):
                cmds = self._cmd.split('\n')  # create list of commands
            elif callable(self._cmd):
                cmds = [self._cmd]  # a function call
            else:
                cmds = self._cmd  # already a list of whatever

            for cmd in cmds:
                if self._out is not None:
                    cmd += ' > %s 2>&1' % self._out
                    # TODO: more general, this only works for bash.

                print(cyan(cmd))

                if self._env.showOnly:
                    continue  # we don't really execute the command here

                if callable(cmd):  # cmd could be a function: call it
                    cmd()
                else:  # if not, it's a command: make a system call
                    call(cmd, shell=True, env=self._environ,
                         stdout=sys.stdout, stderr=sys.stderr)

            # Return to working directory, useful when we change dir
            # before executing the command.
            os.chdir(cwd)
            if not self._env.showOnly:
                for t in self._targets:
                    assert glob(t), ("target '%s' not built (after "
                                     "running '%s')" % (t, cmd))

    def __str__(self):
        return "Command: %s, targets: %s" % (self._cmd, self._targets)


class Target:
    def __init__(self, env, name, *commands, **kwargs):
        self._env = env
        self._name = name
        self._default = kwargs.get('default', False)
        self._always = kwargs.get('always', False)  # Adding always here to allow getting to Commands where always=True
        self._commandList = list(commands)  # copy the list/tuple of commands
        self._finalCommands = []  # their targets will be used to check if we need to re-build
        self._deps = []  # names of dependency targets

    def getCommands(self):
        return self._commandList

    def addCommand(self, cmd, **kwargs):
        if isinstance(cmd, Command):
            c = cmd
        else:
            c = Command(self._env, cmd, **kwargs)
        self._commandList.append(c)

        if kwargs.get('final', False):
            self._finalCommands.append(c)
        return c

    def addDep(self, dep):
        self._deps.append(dep)

    def getDeps(self):
        return self._deps

    def _existsAll(self):
        for c in self._finalCommands:
            if not c._existsAll():
                return False
        return True

    def isDefault(self):
        return self._default

    def setDefault(self, default):
        self._default = default

    def getName(self):
        return self._name

    def execute(self):
        t1 = time.time()

        print(green("Building %s ..." % self._name))
        if not self._always and self._existsAll():
            print("  All targets exist, skipping.")
        else:
            for command in self._commandList:
                command.execute()

        if not self._env.showOnly:
            dt = time.time() - t1
            if dt < 60:
                print(green('Done (%.2f seconds)' % dt))
            else:
                print(green('Done (%d m %02d s)' % (dt / 60, int(dt) % 60)))

    def __str__(self):
        return self._name


class Environment:

    def __init__(self, **kwargs):
        self._targetDict = {}
        self._targetList = []
        # We need a targetList which has the targetDict.keys() in order
        # (OrderedDict is not available in python < 2.7)

        self._packages = {}  # dict of available packages (to show in --help)

        self._args = kwargs.get('args', [])
        self.showOnly = '--show' in self._args

        # Find if the -j arguments was passed to get the number of processors
        if '-j' in self._args:
            j = self._args.index('-j')
            self._processors = int(self._args[j + 1])
        else:
            self._processors = 1

        if LINUX:
            self._libSuffix = 'so'  # Shared libraries extension name
        else:
            self._libSuffix = 'dylib'

        self._downloadCmd = ('wget -nv -c -O %(tar)s.part %(url)s\n'
                             'mv -v %(tar)s.part %(tar)s')
        # Removed the z: "The tar command auto-detects compression type and extracts the archive"
        # From https://linuxize.com/post/how-to-extract-unzip-tar-bz2-file/#extracting-tarbz2-file
        self._tarCmd = 'tar -xf %s'
        self._pipCmd = kwargs.get('pipCmd', 'pip install %s==%s')

    def getLibSuffix(self):
        return self._libSuffix

    def getProcessors(self):
        return self._processors

    @staticmethod
    def getSoftware(*paths):
        return os.path.join(Config.SCIPION_SOFTWARE, *paths)

    @staticmethod
    def getLibFolder(*paths):
        return Environment.getSoftware("lib", *paths)

    @staticmethod
    def getPython():
        return sys.executable

    # Pablo: A quick search didn't find usages.
    # @staticmethod
    # def getPythonFolder():
    #     return Environment.getLibFolder() + '/python2.7'

    @staticmethod
    def getPythonPackagesFolder():
        # This does not work on MAC virtual envs
        # import site
        # return site.getsitepackages()[0]

        from distutils.sysconfig import get_python_lib
        return get_python_lib()

    @staticmethod
    def getIncludeFolder():
        return Environment.getSoftware('include')

    def getLib(self, name):

        return Environment.getLibFolder('lib%s.%s' % (name, self._libSuffix))

    @staticmethod
    def getBinFolder(*paths):
        return os.path.join(mkdir(Environment.getSoftware('bin')), *paths)

    @staticmethod
    def getBin(name):
        return Environment.getBinFolder(name)

    @staticmethod
    def getTmpFolder():
        return mkdir(Environment.getSoftware('tmp'))

    @staticmethod
    def getLogFolder(*path):
        return os.path.join(mkdir(Environment.getSoftware('log')), *path)

    @staticmethod
    def getEmFolder():
        return mkdir(pwem.Config.EM_ROOT)

    @staticmethod
    def getEm(name):
        return '%s/%s' % (Environment.getEmFolder(), name)

    def getTargetList(self):
        return self._targetList

    def addTarget(self, name, *commands, **kwargs):

        if name in self._targetDict:
            raise Exception("Duplicated target '%s'" % name)

        t = Target(self, name, *commands, **kwargs)
        self._targetList.append(t)
        self._targetDict[name] = t

        return t

    def addTargetAlias(self, name, alias):
        """ Add an alias to an existing target.
        This function will be used for installing the last version of each
        package.
        """
        if name not in self._targetDict:
            raise Exception("Can't add alias, target name '%s' not found. "
                            % name)

        self._targetDict[alias] = self._targetDict[name]

    def getTarget(self, name):
        return self._targetDict[name]

    def hasTarget(self, name):
        return name in self._targetDict

    def getTargets(self):
        return self._targetList

    def _addTargetDeps(self, target, deps):
        """ Add the dependencies to target.
        Check that each dependency correspond to a previous target.
        """
        for d in deps:
            if isinstance(d, str):
                targetName = d
            elif isinstance(d, Target):
                targetName = d.getName()
            else:
                raise Exception("Dependencies should be either string or "
                                "Target, received: %s" % d)

            if targetName not in self._targetDict:
                raise Exception("Dependency '%s' does not exists. " % targetName)

            target.addDep(targetName)

    def _addDownloadUntar(self, name, **kwargs):
        """ Buid a basic target and add commands for Download and Untar.
        This is the base for addLibrary, addModule and addPackage.
        """
        # Use reasonable defaults.
        tar = kwargs.get('tar', '%s.tgz' % name)
        urlSuffix = kwargs.get('urlSuffix', 'external')
        url = kwargs.get('url', '%s/%s/%s' % (Config.SCIPION_URL_SOFTWARE, urlSuffix, tar))
        downloadDir = kwargs.get('downloadDir', self.getTmpFolder())
        buildDir = kwargs.get('buildDir',
                              tar.rsplit('.tar.gz', 1)[0].rsplit('.tgz', 1)[0])
        targetDir = kwargs.get('targetDir', buildDir)

        createBuildDir = kwargs.get('createBuildDir', False)

        deps = kwargs.get('deps', [])

        # Download library tgz
        tarFile = join(downloadDir, tar)
        buildPath = join(downloadDir, buildDir)
        targetPath = join(downloadDir, targetDir)

        t = self.addTarget(name, default=kwargs.get('default', True))
        self._addTargetDeps(t, deps)
        t.buildDir = buildDir
        t.buildPath = buildPath
        t.targetPath = targetPath

        # check if tar exists and has size >0 so that we can download again
        if os.path.isfile(tarFile) and os.path.getsize(tarFile) == 0:
            os.remove(tarFile)

        if url.startswith('file:'):
            t.addCommand('ln -s %s %s' % (url.replace('file:', ''), tar),
                         targets=tarFile,
                         cwd=downloadDir)
        else:
            t.addCommand(self._downloadCmd % {'tar': tarFile, 'url': url},
                         targets=tarFile)

        if createBuildDir:
            tarCmd = '{0} -C {1}'.format(self._tarCmd % tar, buildDir)
            t.addCommand('mkdir %s' % buildPath,
                         targets=[buildPath],
                         cwd=downloadDir)
        else:
            tarCmd = self._tarCmd % tar

        finalTarget = join(downloadDir, kwargs.get('target', buildDir))
        t.addCommand(tarCmd,
                     targets=finalTarget,
                     cwd=downloadDir)

        return t

    def addLibrary(self, name, **kwargs):
        """Add library <name> to the construction process.

        Checks that the needed programs are in PATH, needed libraries
        can be found, downloads the given url, untars the resulting
        tar file, configures the library with the given flags,
        compiles it (in the given buildDir) and installs it.

        If default=False, the library will not be built.

        Returns the final targets, the ones that Make will create.

        """
        configTarget = kwargs.get('configTarget', 'Makefile')
        configAlways = kwargs.get('configAlways', False)
        flags = kwargs.get('flags', [])
        targets = kwargs.get('targets', [self.getLib(name)])
        clean = kwargs.get('clean', False)  # Execute make clean at the end??
        cmake = kwargs.get('cmake', False)  # Use cmake instead of configure??
        default = kwargs.get('default', True)
        neededProgs = kwargs.get('neededProgs', [])
        libChecks = kwargs.get('libChecks', [])

        if default or name in sys.argv[2:]:
            # Check that we have the necessary programs and libraries in place.
            for prog in neededProgs:
                assert progInPath(prog), ("Cannot find necessary program: %s\n"
                                          "Please install and try again" % prog)
            for lib in libChecks:
                checkLib(lib)

        # If passing a command list (of tuples (command, target)) those actions
        # will be performed instead of the normal ./configure / cmake + make
        commands = kwargs.get('commands', [])

        t = self._addDownloadUntar(name, **kwargs)
        configDir = kwargs.get('configDir', t.buildDir)

        configPath = join(self.getTmpFolder(), configDir)
        makeFile = '%s/%s' % (configPath, configTarget)
        prefix = abspath(Environment.getSoftware())

        # If we specified the commands to run to obtain the target,
        # that's the only thing we will do.
        if commands:
            for cmd, tgt in commands:
                t.addCommand(cmd, targets=tgt, final=True)
                # Note that we don't use cwd=t.buildDir, so paths are
                # relative to SCIPION_HOME.
            return t

        # If we didn't specify the commands, we can either compile
        # with autotools (so we have to run "configure") or cmake.

        environ = os.environ.copy()
        for envVar, value in [('CPPFLAGS', '-I%s/include' % prefix),
                              ('LDFLAGS', '-L%s/lib' % prefix)]:
            environ[envVar] = '%s %s' % (value, os.environ.get(envVar, ''))

        if not cmake:
            flags.append('--prefix=%s' % prefix)
            flags.append('--libdir=%s/lib' % prefix)
            t.addCommand('./configure %s' % ' '.join(flags),
                         targets=makeFile, cwd=configPath,
                         out=self.getLogFolder('%s_configure.log' % name),
                         always=configAlways, environ=environ)
        else:
            assert progInPath('cmake') or 'cmake' in sys.argv[2:], \
                "Cannot run 'cmake'. Please install it in your system first."

            flags.append('-DCMAKE_INSTALL_PREFIX:PATH=%s .' % prefix)
            t.addCommand('cmake %s' % ' '.join(flags),
                         targets=makeFile, cwd=configPath,
                         out=self.getLogFolder('%s_cmake.log' % name),
                         environ=environ)

        t.addCommand('make -j %d' % self._processors,
                     cwd=t.buildPath,
                     out=self.getLogFolder('%s_make.log' % name))

        t.addCommand('make install',
                     targets=targets,
                     cwd=t.buildPath,
                     out=self.getLogFolder('%s_make_install.log' % name),
                     final=True)

        if clean:
            t.addCommand('make clean',
                         cwd=t.buildPath,
                         out=self.getLogFolder('%s_make_clean.log' % name))
            t.addCommand('rm %s' % makeFile)

        return t

    def addPipModule(self, name, version="", pipCmd=None,
                     target=None, default=True, deps=[]):
        """Add a new module to our built Python .
        Params in kwargs:
            name: pip module name
            version: module version - must be specified to prevent undesired updates.
            default: True if this module is build by default.
        """

        target = name if target is None else target
        pipCmd = pipCmd or self._pipCmd % (name, version)
        t = self.addTarget(name, default=default, always=True)  # we set always=True to let pip decide if updating

        # Add the dependencies
        defaultDeps = []

        self._addTargetDeps(t, defaultDeps + deps)

        t.addCommand(pipCmd,
                     final=True,
                     targets="%s/%s" % (self.getPythonPackagesFolder(), target),
                     always=True  # execute pip command always. Pip will handle target existence
                     )

        return t

    def addPackage(self, name, **kwargs):
        """ Download a package tgz, untar it and create a link in software/em.
        Params in kwargs:
            tar: the package tar file, by default the name + .tgz. Pass None or VOID_TGZ if there is no tar file.
            commands: a list with actions to be executed to install the package
        """
        # Add to the list of available packages, for reference (used in --help).
        neededProgs = kwargs.get('neededProgs', [])

        if name in sys.argv[2:]:
            # Check that we have the necessary programs in place.
            for prog in neededProgs:
                assert progInPath(prog), ("Cannot find necessary program: %s\n"
                                          "Please install and try again" % prog)

        if name not in self._packages:
            self._packages[name] = []

        # Get the version from the kwargs
        if 'version' in kwargs:
            version = kwargs['version']
            extName = self._getExtName(name, version)
        else:
            version = ''
            extName = name

        self._packages[name].append((name, version))

        environ = (self.updateCudaEnviron(name)
                   if kwargs.get('updateCuda', False) else None)

        # Set environment
        variables = kwargs.get('vars', {})
        if variables:
            environ = {} if environ is None else environ
            environ.update(variables)

        # We reuse the download and untar from the addLibrary method
        # and pass the createLink as a new command 
        tar = kwargs.get('tar', '%s.tgz' % extName)

        # If tar is None or void.tgz
        if tar is None or tar == VOID_TGZ:
            tar = VOID_TGZ
            kwargs["buildDir"] = extName
            kwargs["createBuildDir"] = True

        buildDir = kwargs.get('buildDir',
                              tar.rsplit('.tar.gz', 1)[0].rsplit('.tgz', 1)[0])
        targetDir = kwargs.get('targetDir', buildDir)

        libArgs = {'downloadDir': self.getEmFolder(),
                   'urlSuffix': 'em',
                   'default': False}  # This will be updated with value in kwargs
        libArgs.update(kwargs)

        target = self._addDownloadUntar(extName, **libArgs)
        commands = kwargs.get('commands', [])
        for cmd, tgt in commands:
            if isinstance(tgt, basestring):
                tgt = [tgt]

            # Take all package targets relative to package build dir
            normTgt = []
            for t in tgt:
                # Check for empty targets and warn about them
                if not t:
                    print("WARNING: Target empty for command %s" % cmd)

                normTgt.append(join(target.targetPath, t))

            target.addCommand(cmd, targets=normTgt, cwd=target.buildPath,
                              final=True, environ=environ)

        target.addCommand(Command(self, Link(extName, targetDir),
                                  targets=[self.getEm(extName),
                                           self.getEm(targetDir)],
                                  cwd=self.getEm('')),
                          final=True)

        # Create an alias with the name for that version
        # this imply that the last package version added will be
        # the one installed by default, so the last versions should
        # be the last ones to be inserted
        self.addTargetAlias(extName, name)

        return target

    def _showTargetGraph(self, targetList):
        """ Traverse the targets taking into account
        their dependencies and print them in DOT format.
        """
        print('digraph libraries {')
        for tgt in targetList:
            deps = tgt.getDeps()
            if deps:
                print('\n'.join("  %s -> %s" % (tgt, x) for x in deps))
            else:
                print("  %s" % tgt)
        print('}')

    def _showTargetTree(self, targetList, maxLevel=-1):
        """ Print the tree of dependencies for the given targets,
        up to a depth level of maxLevel (-1 for unlimited).
        """
        # List of (indent level, target)
        nodes = [(0, tgt) for tgt in targetList[::-1]]
        while nodes:
            lvl, tgt = nodes.pop()
            print("%s- %s" % ("  " * lvl, tgt))
            if maxLevel != -1 and lvl >= maxLevel:
                continue
            nodes.extend((lvl + 1, self._targetDict[x]) for x in tgt.getDeps())

    def _executeTargets(self, targetList):
        """ Execute the targets in targetList, running all their
        dependencies first.
        """
        executed = set()  # targets already executed
        exploring = set()  # targets whose dependencies we are exploring
        targets = targetList[::-1]
        while targets:
            tgt = targets.pop()
            if tgt.getName() in executed:
                continue
            deps = tgt.getDeps()
            if set(deps) - executed:  # there are dependencies not yet executed
                if tgt.getName() in exploring:
                    raise RuntimeError("Cyclic dependency on %s" % tgt)
                exploring.add(tgt.getName())
                targets.append(tgt)
                targets.extend(self._targetDict[x] for x in deps)
            else:
                tgt.execute()
                executed.add(tgt.getName())
                exploring.discard(tgt.getName())

    @staticmethod
    def _getExtName(name, version):
        """ Return folder name for a given package-version """
        return '%s-%s' % (name, version)

    def _isInstalled(self, name, version):
        """ Return true if the package-version seems to be installed. """
        pydir = self.getPythonPackagesFolder()
        extName = self._getExtName(name, version)
        return (exists(join(self.getEmFolder(), extName)) or
                extName in [x[:len(extName)] for x in os.listdir(pydir)])

    def printHelp(self):
        printStr = ""
        if self._packages:
            printStr = ("Available binaries: "
                        "([ ] not installed, [X] seems already installed)\n\n")

            keys = sorted(self._packages.keys())
            for k in keys:
                pVersions = self._packages[k]
                printStr += "{0:25}".format(k)
                for name, version in pVersions:
                    installed = self._isInstalled(name, version)
                    printStr += '{0:8}[{1}]{2:5}'.format(version, 'X' if installed else ' ', ' ')
                printStr += '\n'
        return printStr

    def execute(self):
        if '--help' in self._args:
            print(self.printHelp())
            return

        # Check if there are explicit targets and only install
        # the selected ones, ignore starting with 'xmipp'
        cmdTargets = [a for a in self._args
                      if a[0].isalpha()]
        if cmdTargets:
            # Check that they are all command targets
            for t in cmdTargets:
                if t not in self._targetDict:
                    raise RuntimeError("Unknown target: %s" % t)
            # Grab the targets passed in the command line
            targetList = [self._targetDict[t] for t in cmdTargets]
        else:
            # use all targets marked as default
            targetList = [t for t in self._targetList if t.isDefault()]

        if '--show-tree' in self._args:
            if '--dot' in self._args:
                self._showTargetGraph(targetList)
            else:
                self._showTargetTree(targetList)
        else:
            self._executeTargets(targetList)

    def updateCudaEnviron(self, package):
        """ Update the environment adding CUDA_LIB and/or CUDA_BIN to support
        packages that uses CUDA.
        package: package that needs CUDA to compile.
        """
        packUpper = package.upper()
        cudaLib = os.environ.get(packUpper + '_CUDA_LIB')
        cudaBin = os.environ.get(packUpper + '_CUDA_BIN')

        if cudaLib is None:
            cudaLib = pwem.Config.CUDA_LIB
            cudaBin = pwem.Config.CUDA_BIN

        environ = os.environ.copy()

        # If there isn't any CUDA in the environment
        if cudaLib is None and cudaBin is None:
            # Exit ...do not update the environment
            return environ

        elif cudaLib is not None and cudaBin is None:
            raise Exception("CUDA_LIB (or %s_CUDA_LIB) is defined, but not "
                            "CUDA_BIN (or %s_CUDA_BIN), please execute "
                            "scipion config --update" % (packUpper, packUpper))
        elif cudaBin is not None and cudaLib is None:
            raise Exception("CUDA_BIN (or %s_CUDA_BIN) is defined, but not "
                            "CUDA_LIB (or %s_CUDA_LIB), please execute "
                            "scipion config --update" % (packUpper, packUpper))
        elif os.path.exists(cudaLib) and os.path.exists(cudaBin):
            environ.update({'LD_LIBRARY_PATH': cudaLib + ":" +
                                               environ['LD_LIBRARY_PATH']})
            environ.update({'PATH': cudaBin + ":" + environ['PATH']})

        return environ

    def setDefault(self, default):
        """Set default values of all packages to the passed parameter"""
        for t in self._targetList:
            t.setDefault(default)

    def getPackages(self):
        """Return all plugin packages"""
        return self._packages

    def hasPackage(self, name):
        """ Returns true if it has the package"""
        return name in self._packages

    def getPackage(self, name):
        return self._packages.get(name, None)


class Link:
    def __init__(self, packageLink, packageFolder):
        self._packageLink = packageLink
        self._packageFolder = packageFolder

    def __call__(self):
        self.createPackageLink(self._packageLink, self._packageFolder)

    def __str__(self):
        return "Link '%s -> %s'" % (self._packageLink, self._packageFolder)

    def createPackageLink(self, packageLink, packageFolder):
        """ Create a link to packageFolder in packageLink, validate
        that packageFolder exists and if packageLink exists it is 
        a link.
        This function is supposed to be executed in software/em folder.
        """
        linkText = "'%s -> %s'" % (packageLink, packageFolder)

        if not exists(packageFolder):
            print(red("Creating link %s, but '%s' does not exist!!!\n"
                      "INSTALLATION FAILED!!!" % (linkText, packageFolder)))
            sys.exit(1)

        if exists(packageLink):
            if islink(packageLink):
                os.remove(packageLink)
            else:
                print(red("Creating link %s, but '%s' exists and is not a link!!!\n"
                          "INSTALLATION FAILED!!!" % (linkText, packageLink)))
                sys.exit(1)

        os.symlink(packageFolder, packageLink)
        print("Created link: %s" % linkText)


def mkdir(path):
    """ Creates a folder if it does not exists"""
    if not exists(path):
        os.makedirs(path)
    return path
