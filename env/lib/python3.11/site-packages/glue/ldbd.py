# -*- coding: utf-8
#
# This file is part of the Grid LSC User Environment (GLUE)
#
# GLUE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

"""lightweight database dumper
Copyright (C) 2003 Duncan Brown
This file is part of the lightweight datapase dumper (ldbd)

The ldbd module provides classes for manipulating LIGO metadata database
tables.

References:
http://www.ligo.caltech.edu/docs/T/T990101-02.pdf
http://www.ligo.caltech.edu/docs/T/T990023-01.pdf
http://ldas-sw.ligo.caltech.edu/doc/db2/doc/html/ilwdformat.html
"""

import csv
import re

from glue import git_version

__author__ = 'Duncan Brown <dbrown@ligo.caltech.edu>'
__date__ = git_version.date
__version__ = git_version.id


class LIGOLWStream(csv.Dialect):
  """
  Create a csv parser dialect for parsing LIGO_LW streams
  """
  delimiter = ','
  doublequote = False
  escapechar = '\\'
  lineterminator = '\n'
  quotechar = '"'
  quoting = csv.QUOTE_ALL
  skipinitialspace = True

csv.register_dialect("LIGOLWStream",LIGOLWStream)


class LIGOLwParseError(Exception):
  """Error parsing LIGO lightweight XML file"""
  pass


class Xlator(dict):
  """
  All in one multiple string substitution class from the python cookbook
  """
  def _make_regex(self):
    """
    Build a re object based on keys in the current dictionary
    """
    return re.compile("|".join(map(re.escape, self.keys())))

  def __call__(self, match):
    """
    Handler invoked for each regex match
    """
    return self[match.group(0)]

  def xlat(self, text):
    """
    Translate text, returns the modified text
    """
    return self._make_regex().sub(self,text)


class LIGOLwParser:
  """
  Provides methods for parsing the data from a LIGO lightweight XML
  file parsed with pyRXP into a dictionary
  """

  def __init__(self):
    """
    Initializes a LIGO lightweight XML parser with the necessary
    regular expressions and function for tuple translation
    """
    self.tabrx = re.compile(r'(\A[a-z0-9_]+:|\A)([a-z0-9_]+):table\Z')
    self.colrx = re.compile(r'(\A[a-z0-9_]+:|\A)([a-z0-9_]+:|\A)([a-z0-9_]+)\Z')
    self.llsrx = re.compile(r'\A\s*"')
    self.rlsrx = re.compile(r'"\s*\Z')
    self.licrx = re.compile(r'\A\s+"')
    self.ricrx = re.compile(r'"*\s*\Z')
    self.octrx = re.compile(r'\A\\[0-9][0-9][0-9]')
    self.dlmrx = re.compile(r'\\,')
    self.unique = None
    self.types = {
      'int2s' : int,
      'int_2s' : int,
      'int4s' : int,
      'int_4s' : int,
      'int8s' : int,
      'int_8s' : int,
      'real4' : float,
      'real_4' : float,
      'real8' : float,
      'real_8' : float,
      'lstring' : self.__lstring,
      'ilwd:char' : self.__ilwdchar,
      'ilwd:char_u' : self.__ilwdchar
    }
    self.xmltostr = Xlator({ r'&amp;' : r'&', r'&gt;' : r'>', r'&lt;' : r'<','\\\\' : '\\'}) # Note: see https://www.gravity.phy.syr.edu/dokuwiki/doku.php?id=rpfisher:gluebughunt if this is confusing, the parser just cleanly handles the conversion of everything

  def __lstring(self,lstr):
    """
    Returns a parsed lstring by stripping out and instances of
    the escaped delimiter. Sometimes the raw lstring has whitespace
    and a double quote at the beginning or end. If present, these
    are removed.
    """
    lstr = self.llsrx.sub('',lstr)
    lstr = self.rlsrx.sub('',lstr)
    lstr = self.xmltostr.xlat(lstr)
    lstr = self.dlmrx.sub(',',lstr)
    return lstr

  def __ilwdchar(self,istr):
    """
    If the ilwd:char field contains octal data, it is translated
    to a binary string and returned. Otherwise a lookup is done
    in the unique id dictionary and a binary string containing the
    correct unique id is returned.
    """
    istr_orig = istr
    istr = self.licrx.sub('',istr)
    istr = self.ricrx.sub('',istr)
    if self.octrx.match(istr):
      exec("istr = '"+istr+"'")
    else:
      try:
        istr = self.unique.lookup(istr)
      except AttributeError:
        if not self.unique:
          istr = istr_orig
        else:
          raise LIGOLwParseError('unique id table has not been initialized')
    return istr

  def parsetuple(self,xmltuple):
    """
    Parse an XML tuple returned by pyRXP into a dictionary
    of LIGO metadata elements. The dictionary contains one
    entry for each table found in the XML tuple.
    """
    # first extract all the table and columns from the tuple from the
    # children of the ligo lightweight parent tuple
    table = {}
    tupleidx = 0
    for tag in xmltuple[2]:
      if tag[0] == 'Table' or tag[0] == 'table':
        tab = tag[1]['Name'].lower()
        try:
          tab = self.tabrx.match(tab).group(2)
        except AttributeError:
          raise LIGOLwParseError('unable to parse a valid table name '+tab)
        # initalize the table dictionary for this table
        table[tab] = {
          'pos' : tupleidx,
          'column' : {},
          'stream' : (),
          'query' : ''
          }
        # parse for columns in the tables children
        # look for the column name and type in the attributes
        # store the index in which the columns were found as
        # we need this to decode the stream later
        for subtag in tag[2]:
          if subtag[0] == 'Column' or subtag[0] == 'column':
            col = subtag[1]['Name'].lower()
            try:
              col = self.colrx.match(col).group(3)
            except AttributeError:
              raise LIGOLwParseError('unable to parse a valid column name '+col)
            try:
              typ = subtag[1]['Type'].lower()
            except KeyError:
              raise LIGOLwParseError('type is missing for column '+col)
            table[tab]['column'][col] = typ
            table[tab].setdefault('orderedcol',[]).append(col)
      tupleidx += 1

    # now iterate the dictionary of tables we have created looking for streams
    for tab in table.keys():
      for tag in xmltuple[2][table[tab]['pos']][2]:
        if tag[0] == 'Stream' or tag[0] == 'stream':
          # store the stream delimiter and create the esacpe regex
          try:
            delim = tag[1]['Delimiter']
          except KeyError:
            raise LIGOLwParseError('stream is missing delimiter')
          if delim != ',':
            raise LIGOLwParseError('unable to handle stream delimiter: '+delim)

          # If the result set is empty tag[2] is an empty array, which causes
          # the next step to fail.  Add an empty string in this case.
          if len(tag[2]) == 0:
            tag[2].append("")

          # if the final row of the table ends with a null entry, there will
          # be an extra delimeter at the end of the stream,
          # so we need to strip that out;
          # see glue/ligolw/table.py#L469-L471 (as of glue-release-2.0.0)
          raw = tag[2][0]
          if raw.endswith(delim+delim):
              raw = raw[:-len(delim)]

          # strip newlines from the stream and parse it
          stream = next(csv.reader([re.sub(r'\n','',raw)],LIGOLWStream))

          # turn the csv stream into a list of lists
          slen = len(stream)
          ntyp = len(table[tab]['column'])
          mlen, lft = divmod(slen,ntyp)
          if lft != 0:
            raise LIGOLwParseError('invalid stream length for given columns')
          lst = [[None] * ntyp for i in range(mlen)]

          # translate the stream data to the correct data types
          for i in range(slen):
            j, k = divmod(i,ntyp)
            try:
              thiscol = table[tab]['orderedcol'][k]
              if len( stream[i] ) == 0:
                lst[j][k] = None
              else:
                lst[j][k] = self.types[table[tab]['column'][thiscol]](stream[i])
            except (KeyError, ValueError) as errmsg:
              msg = "stream translation error (%s) " % str(errmsg)
              msg += "for column %s in table %s: %s -> %s" \
                % (tab,thiscol,stream[i],str(table[tab]))
              raise LIGOLwParseError(msg)
          table[tab]['stream'] = list(map(tuple,lst))

    # return the created table to the caller
    return table


class LIGOMetadata:
  """
  LIGO Metadata object class. Contains methods for parsing a LIGO
  lightweight XML file and inserting it into a database, executing
  and SQL query to retrive data from the database and writing it
  to a LIGO lightweight XML file
  """
  def __init__(self,xmlparser=None,lwtparser=None):
    """
    Connects to the database and creates a cursor. Initializes the unique
    id table for this LIGO lw document.

    xmlparser = pyRXP XML to tuple parser object
    lwtparser = LIGOLwParser object (tuple parser)
    """
    self.xmlparser = xmlparser
    self.lwtparser = lwtparser
    if lwtparser:
      self.lwtparser.unique = None
    self.table = {}
    self.strtoxml = Xlator({ r'&' : r'&amp;', r'>' : r'&gt;', r'<' : r'&lt;', '\\' : '\\\\', '\"' : '\\\"' }) # Note: see https://www.gravity.phy.syr.edu/dokuwiki/doku.php?id=rpfisher:gluebughunt if this is confusing, the parser just cleanly handles the conversion of everything

  def parse(self,xml):
    """
    Parses an XML document into a form read for insertion into the database

    xml = the xml document to be parsed
    """
    if not self.xmlparser:
      raise LIGOLwParseError("pyRXP parser not initialized")
    if not self.lwtparser:
      raise LIGOLwParseError("LIGO_LW tuple parser not initialized")
    xml = "".join([x.strip() for x in xml.split('\n')])
    ligolwtup = self.xmlparser(xml.encode("utf-8"))
    self.table = self.lwtparser.parsetuple(ligolwtup)
