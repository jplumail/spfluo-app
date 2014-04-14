# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This module is mainly for the Viewer class, which 
serve as base for implementing visualization tools(Viewer sub-classes).
"""

from os.path import join
from protocol import Protocol

DESKTOP_TKINTER = 'tkinter'
WEB_DJANGO = 'django'


class View(object):
    """ Represents a visualization result for some object or file.
    Views can be plots, table views, chimera scripts, commands or messages.
    """
    def show(self):
        """ This method should be overriden to implement how
        this particular view will be displayed in desktop.
        """
        pass
    
    def toUrl(self):
        """ If the view have web implementation, this method
        should be implented to build the url with parameters
        that will be used to respond.
        """
        pass
    
    
class Viewer(object):
    """ A Viewer will provide several Views to visualize
    the data associated to data objects or protocol.
    
    The _targets class property should contains a list of string
    with the class names that this viewer is able to visualize.
    For example: _targets = ['Image', 'SetOfImages']
    """
    _targets = []
    _environments = [DESKTOP_TKINTER]
    
    def __init__(self, tmpPath='./Tmp', **args):
        self._tmpPath = tmpPath
        self._project = args.get('project', None)
        self.protocol = args.get('protocol', None)
        self._parent = args.get('parent', None)
        
    def _getTmpPath(self, *paths):
        return join(self._tmpPath, *paths)
    
    def visualize(self, obj):
        """ This method should make the necessary convertions
        and call the command line utilities to visualize this
        particular object.
        """
        pass
    
    def getView(self):
        """ This method should return the string value of the view in web
        that will respond to this viewer. This method only should be implemented
        in those viewers that have WEB_DJANGO environment defined. 
        """
        return None
    
    def getProject(self):
        return self._project
    
    def setProject(self, project):
        self._project = project
        
    def getParent(self):
        """ Get the Tk parent widget. """
        return self._parent
    
    
class ProtocolViewer(Protocol, Viewer):
    """ Special kind of viewer that have a Form to organize better
    complex visualization associated with protocol results.
    If should provide a mapping between form params and the corresponding
    functions that will return the corresponding Views.
    """
    def __init__(self, **args):
        Protocol.__init__(self, **args)
        Viewer.__init__(self, **args)
        self.allowHeader.set(False)
        self.showPlot = True # This flag will be used to display a plot or return the plotter
        
        
    def setProtocol(self, protocol):
        self.protocol = protocol
    
    def visualize(self, obj, **args):
        """Open the Protocol GUI Form given a Protocol instance"""
        from gui.form import FormWindow
        self.setProtocol(obj)
        self.windows = args.get('windows', None)
        self.formWindow = FormWindow("Protocol Viewer: " + self.getClassName(), self, 
                       self._viewAll, self.windows,
                       visualizeDict=self._getVisualizeDict(),
                       visualizeMode=True)
        self.formWindow.visualizeMode = True
        self.showInfo = self.formWindow.showInfo
        self.showError = self.formWindow.showError
        self.formWindow.show(center=True)     

    def _showPlots(self, plots, errors):
        if len(errors):
            self.showError('\n'.join(errors))
        if len(plots):
            plots[0].show() # Show from any plot, 0 or any other
            
    def _showOrReturn(self, xplotter):
        if self.showPlot:
            xplotter.show()
        else:
            return xplotter
        
    def _getVisualizeDict(self):
        """ Create the visualization dict for view individual params. """
        return {}
    
    def _viewAll(self, *args):
        """ Visualize all data give the parameters. """
        for k, v in self._getVisualizeDict().iteritems():
            print "k: %s, v: %s" % (k, v)
            if self.getAttributeValue(k, False):
                print "   calling v..."
                v(k)
    
    #TODO: This method should not be necessary, instead NumericListParam should return a list and not a String 
    def _getListFromRangeString(self, rangeStr):
        ''' Create a list of integer from a string with range definitions
        Examples:
        "1,5-8,10" -> [1,5,6,7,8,10]
        "2,6,9-11" -> [2,6,9,10,11]
        "2 5, 6-8" -> [2,5,6,7,8]
        '''
        elements = rangeStr.split(',')
        values = []
        for e in elements:
            if '-' in e:
                limits = e.split('-')
                values += range(int(limits[0]), int(limits[1])+1)
            else:
                # If values are separated by comma also splitted 
                values += map(int, e.split())
        return values
