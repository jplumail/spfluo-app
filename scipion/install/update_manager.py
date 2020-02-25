#!/usr/bin/env python
# **************************************************************************
# *
# * Authors:    J. Jimenez de la Morena (jjimenez@cnb.csic.es)
# *
# *  [1] SciLifeLab, Stockholm University
# *  [2] Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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

import argparse
import optparse
from os import environ

from pip._internal.commands.list import ListCommand
import pip._internal.utils.misc as piputils
from pip._internal.commands import create_command


class UpdateManager:

    pluginName = 'scipion-app'

    @classmethod
    def runUpdateManager(cls, args):
        # create the top-level parser
        parser = argparse.ArgumentParser(prog=args[1:],
                                         formatter_class=argparse.RawTextHelpFormatter)
        subparsers = parser.add_subparsers()
        # create the parser for the "checkupdates" command
        parser_f = subparsers.add_parser('checkupdates',
                                         description='description: check for Scipion updates.',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         usage="{} [-h/--help] [-f/--forceupdate]".format(' '.join(args[:2]))
                                         )
        parser_f.add_argument('-f', '--forceupdate',
                              help='force to update Scipion in case a new update is available',
                              action="store_true")

        parsedArgs = parser.parse_args(args[1:])
        if not cls.isScipionUpToDate():
            print('Scipion is up to date.')
        else:
            print('A new update is available for Scipion.')
            if parsedArgs.forceupdate():
                cls.updateScipion()
            else:
                print('\nYou can update it if you wish in two ways from a terminal in Scipion3 environment:',
                      '\n\t1) Executing pip install scipion-app --upgrade',
                      '\n\t2) Forcing the update from Scipion: {} checkupdates -f/--forceupdate\n'.format(
                          environ.get('SCIPION_HOME', '[path_to_scipion]')))

    @classmethod
    def getUpToDatePluginList(cls):
        return [x.project_name for x in cls.getUpToDatePackages()]

    @classmethod
    def isScipionUpToDate(cls):
        return cls.pluginName in cls.getUpToDatePluginList()

    @classmethod
    def getUpToDatePackages(cls):
        options = optparse.Values({
            'skip_requirements_regex': '',
            'retries': 5, 'pre': False,
            'version': None,
            'include_editable': True,
            'disable_pip_version_check': False,
            'log': None,
            'trusted_hosts': [],
            'outdated': False,
            'no_input': False,
            'local': False,
            'timeout': 15,
            'proxy': '',
            'uptodate': True,
            'help': None,
            'cache_dir': '',
            'no_color': False,
            'user': False,
            'client_cert': None,
            'quiet': 0,
            'not_required': None,
            'no_python_version_warning': False,
            'extra_index_urls': [],
            'isolated_mode': False,
            'exists_action': [],
            'no_index': False,
            'index_url': 'https://pypi.org/simple',
            'find_links': [],
            'path': None,
            'require_venv': False,
            'list_format': 'columns',
            'editable': False,
            'verbose': 0,
            'cert': None})
        distributions = piputils.get_installed_distributions(
            local_only=options.local,
            user_only=options.user,
            editables_only=options.editable,
            include_editables=options.include_editable,
            paths=options.path)
        return cls.genListCommand().get_uptodate(distributions, options)

    @classmethod
    def updateScipion(cls):
        kwargs = {'isolated': False}
        cmd_args = [cls.pluginName,
                    '--upgrade',
                    '-qqq']  # -q, --quiet Give less output. Option is additive, and can be used up to 3 times
        # (corresponding to WARNING, ERROR, and CRITICAL logging levels).

        command = create_command('install', **kwargs)
        status = command.main(cmd_args)
        if status == 0:
            print('Scipion was correctly updated.')
        else:
            print('Something went wrong during the update. Retrying...')
            # Re-launch the update command, so the highest level of verbosity is used to explain the user the problem
            # found
            cmd_args[2] = '-vvv'
            command.main(cmd_args)

    @staticmethod
    def genListCommand():
        _args = ()
        _kw = {'summary': 'List installed packages.', 'name': 'list', 'isolated': False}
        return ListCommand(*_args, **_kw)


