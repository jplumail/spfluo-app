# **************************************************************************
# *
# * Authors:     Grigory Sharov     (sharov@igbmc.fr)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
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

from pyworkflow.protocol.params import (PointerParam, FloatParam, StringParam,
                                        BooleanParam, IntParam, LEVEL_ADVANCED)
import pyworkflow.em.metadata as md
from pyworkflow.utils import exists
from pyworkflow.em.protocol import EMProtocol
from convert import convertBinaryVol, readSetOfParticles, writeSetOfParticles, writeReferences




class ProtRelionSort(EMProtocol):
    """
    Relion particle sorting protocol.
    It calculates difference images between particles and their aligned (and CTF-convoluted)
    references, and calculate Z-score on the characteristics of these difference images (such
    as mean, standard deviation, skewness, excess kurtosis and rotational symmetry).

    """
    _label = 'sort particles'

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
    
    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles', pointerCondition='hasAlignment',
                      label='Input particles',
                      help='Select a set of particles that were previously matched against the references. '
                           'Input should have 2D alignment parameters and class assignment. '
                           'It can be particles after 2D/3D classification, 3D refinement or '
                           'reference-based auto-picking.')
        form.addParam('inputReferences', PointerParam,
                      pointerClass="SetOfClasses2D, SetOfClasses3D, Volume",
                      label='Input references',
                      help='Select references: a set of classes or a 3D volume')
        form.addParam('maskDiameterA', IntParam, default=-1,
                      label='Particle mask diameter (A)',
                      help='The experimental images will be masked with a soft circular mask '
                           'with this <diameter>. '
                           'Make sure this diameter is not set too small because that may mask '
                           'away part of the signal! If set to a value larger than the image '
                           'size no masking will be performed.\n\n'
                           'The same diameter will also be used for a spherical mask of the '
                           'reference structures if no user-provided mask is specified.')
        form.addParam('doLowPass', IntParam, default=-1,
                      label='Low pass filter references to (A):',
                      help='Lowpass filter in Angstroms for the references (prevent Einstein-from-noise!)')
        form.addParam('doInvert', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Invert contrast of references?',
                      help='Density in particles is inverted compared to the density in references')
        form.addParam('doCTF', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Do CTF-correction?',
                      help='If set to Yes, CTFs will be corrected inside the MAP refinement. '
                           'The resulting algorithm intrinsically implements the optimal linear, '
                           'or Wiener filter. Note that input particles should contains CTF parameters.')
        form.addParam('ignoreCTFUntilFirstPeak', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED, condition='doCTF',
                      label='Ignore CTFs until their first peak?',
                      help='If set to Yes, then CTF-amplitude correction will only be performed from the first peak '
                           'of each CTF onward. This can be useful if the CTF model is inadequate at the lowest resolution. '
                           'Still, in general using higher amplitude contrast on the CTFs (e.g. 10-20%) often yields better results. '
                           'Therefore, this option is not generally recommended.')
        form.addParam('minZ', FloatParam, default=0, expertLevel=LEVEL_ADVANCED,
                      label='Min Z-value?',
                      help='Minimum Z-value to count in the sorting of outliers')
        form.addParam('extraParams', StringParam, default='',
                      label='Additional parameters',
                      help="In this box command-line arguments may be provided that "
                           "are not generated by the GUI. This may be useful for testing "
                           "developmental options and/or expert use of the program, e.g: \n"
                           "--verb 1\n")

        form.addParallelSection(threads=0, mpi=1)
            
    #--------------------------- INSERT steps functions ------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep',
                                 self.inputParticles.get().getObjId(),
                                 self.inputReferences.get().getObjId())
        self._insertRelionStep()
        self._insertFunctionStep('createOutputStep')

    def _insertRelionStep(self):
        """ Prepare the command line arguments before calling Relion. """
        # Join in a single line all key, value pairs of the args dict
        args = {}
        self._setArgs(args)
        params = ' '.join(['%s %s' % (k, str(v)) for k, v in args.iteritems()])

        if self.extraParams.hasValue():
            params += ' ' + self.extraParams.get()

        self._insertFunctionStep('runRelionStep', params)

    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self, particlesId, referencesId):
        """ Create the input file in STAR format as expected by Relion.
        If the input particles comes from Relion, just link the file.
        Params:
            particlesId: use this parameters just to force redo of convert if
                the input particles are changed.
        """
        imgSet = self.inputParticles.get()
        refSet = self.inputReferences.get()
        imgStar = self._getPath('input_particles.star')

        # Pass stack file as None to avoid write the images files
        self.info("Converting set from '%s' into '%s'" % (imgSet.getFileName(), imgStar))
        writeSetOfParticles(imgSet, imgStar, self._getExtraPath())

        if refSet.getClassName() == 'Volume':
            refVol = convertBinaryVol(refSet, self._getTmpPath())
        else:
            self.info("Converting set from '%s' into input_references.star" % refSet.getFileName())
            writeReferences(refSet, self._getExtraPath('input_references'))

    def runRelionStep(self, params):
        """ Execute the relion steps with given params. """
        self.runJob(self._getProgram(), params)
        
    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        sortedImgSet = self._createSetOfParticles()
        sortedImgSet.copyInfo(imgSet)
        readSetOfParticles(self._getPath('sorted_particles.star'),
                           sortedImgSet,
                           alignType=imgSet.getAlignment())

        self._defineOutputs(outputParticles=sortedImgSet)
        self._defineSourceRelation(imgSet, sortedImgSet)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []

        return errors
    
    def _summary(self):
        summary = []

        return summary
    
    #--------------------------- UTILS functions -------------------------------
    def _setArgs(self, args):

        maskDiameter = self.maskDiameterA.get()
        if maskDiameter <= 0:
            x, _, _ = self.inputParticles.get().getDim()
            maskDiameter = self.inputParticles.get().getSamplingRate() * x

        vol = self.getInputReferences.get().getFileName()

        if exists(self._getPath('input_references.star')):
            args.update({'--ref': self._getPath('input_references.star')})

        elif exists(vol.endswith('.mrc')):
            args.update({'--ref': vol})

        args.update({'--i': self._getPath('input_particles.star'),
                     '--particle_diameter': maskDiameter,
                     '--angpix': self.inputParticles.get().getSamplingRate(),
                     '--min_z': self.minZ.get(),
                     '--o': self._getPath('sorted_particles.star')
                     })

        if self.doInvert:
            args[' --invert'] = ''

        if self.ignoreCTFUntilFirstPeak:
            args['--ctf_intact_first_peak'] = ''

        if self.doCTF:
            args['--ctf'] = ''

    def _getProgram(self, program='relion_refine'):
        """ Get the program name depending on the MPI use or not. """
        if self.numberOfMpi > 1:
            program += '_mpi'
        return program
