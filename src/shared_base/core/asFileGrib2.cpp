/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asFileGrib2.h"

asFileGrib2::asFileGrib2(const wxString &FileName, const ListFileMode &FileMode)
        : asFile(FileName, FileMode),
          m_filtPtr(NULL),
          m_index(asNOT_FOUND)
{
    switch (FileMode) {
        case (ReadOnly):
            // OK
            break;
        case (Write):
        case (Replace):
        case (New):
        case (Append):
        default :
            asThrowException(_("Grib files edition is not implemented."));
    }
}

asFileGrib2::~asFileGrib2()
{
    Close();
}

bool asFileGrib2::Open()
{
    if (!Find())
        return false;

    // Let GDAL open the dataset
    if (!OpenDataset())
        return false;

    m_opened = true;

    return true;
}

bool asFileGrib2::Close()
{
    if (m_filtPtr != NULL) {
        m_filtPtr = NULL;
    }
    return true;
}

bool asFileGrib2::OpenDataset()
{
    // Filepath
    wxString filePath = m_fileName.GetFullPath();

    // Open file
    m_filtPtr = fopen(filePath.mb_str(), "r");

    if (m_filtPtr == NULL) // Failed
    {
        wxLogError(_("The opening of the grib file failed."));
        wxFAIL;
        return false;
    }

    // Parse structure
    return ParseStructure();
}

/*
 * See http://stackoverflow.com/questions/11767169/using-libgrib2c-in-c-application-linker-error-undefined-reference-to
 */
bool asFileGrib2::ParseStructure()
{
    g2int currentMessageSize(1);
    g2int seekPosition(0);
    g2int offset(0);
    g2int seekLength(32000);

    for (;;) {
        // Searches a file for the next GRIB message.
        seekgb(m_filtPtr, seekPosition, seekLength, &offset, &currentMessageSize);
        if (currentMessageSize == 0)
            break;    // end loop at EOF or problem

        // Reposition stream position indicator
        if (fseek(m_filtPtr, offset, SEEK_SET) != 0) {
            wxLogError(_("Grib file read error."));
            return false;
        }

        // Read block of data from stream
        unsigned char *cgrib = (unsigned char *) malloc((size_t) currentMessageSize);
        fread(cgrib, sizeof(unsigned char), currentMessageSize, m_filtPtr);
        seekPosition = offset + currentMessageSize;

        // Get the number of gridded fields and the number (and maximum size) of Local Use Sections.
        g2int listSec0[3], listSec1[13], numFields, numLocal;
        g2int ierr = g2_info(cgrib, listSec0, listSec1, &numFields, &numLocal);
        wxASSERT(numFields == 1);
        wxASSERT(numLocal == 0);
        if (ierr > 0) {
            handleGribError(ierr);
            return false;
        }

        for (long n = 0; n < numFields; n++) {

            // Store elements in vectors here in order to ensure a corresponding lengths.
            m_messageOffsets.push_back(offset);
            m_messageSizes.push_back(currentMessageSize);
            m_fieldNum.push_back(n);
            m_refTimes.push_back(asTime::GetMJD((int) listSec1[5], (int) listSec1[6], (int) listSec1[7],
                                                (int) listSec1[8], (int) listSec1[9]));

            // Get all the metadata for a given data field
            gribfield *gfld = NULL;
            int unpack = 0;
            g2int expand = 0;
            ierr = g2_getfld(cgrib, n + 1, unpack, expand, &gfld);
            if (ierr > 0) {
                handleGribError(ierr);
                return false;
            }

            wxASSERT(gfld);

            m_times.push_back(asTime::GetMJD((int) gfld->idsect[5], (int) gfld->idsect[6], (int) gfld->idsect[7],
                                             (int) gfld->idsect[8], (int) gfld->idsect[9]));

            // Grid Definition
            if (!CheckGridDefinition(gfld)) {
                wxLogError(_("Grid definition not allowed yet."));
                return false;
            }

            BuildAxes(gfld);

            // Product Definition
            if (!CheckProductDefinition(gfld)) {
                wxLogError(_("Product definition not allowed yet."));
                return false;
            }

            m_parameterDisciplines.push_back(0);
            m_parameterCategories.push_back((int) gfld->ipdtmpl[0]); // www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-1.shtml
            m_parameterNums.push_back((int) gfld->ipdtmpl[1]); // www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-2-0-3.shtml
            m_forecastTimes.push_back(gfld->ipdtmpl[8]);
            GetLevel(gfld);

            g2_free(gfld);
        }
        free(cgrib);
    }

    // Check unique time value
    for (int i = 0; i < m_times.size(); ++i) {
        if (m_times[i] != m_times[0]) {
            wxLogError(_("Handling of multiple time values in a Grib file is not yet implemented."));
            return false;
        }
    }

    return true;
}

void asFileGrib2::GetLevel(const gribfield *gfld)
{
    float surfVal = gfld->ipdtmpl[11];

    // Type of first fixed surface (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)
    if (gfld->ipdtmpl[9] == 100 || gfld->ipdtmpl[9] == 108) {
        surfVal /= 100; // Pa to hPa
    }

    m_levelTypes.push_back(gfld->ipdtmpl[9]);
    m_levels.push_back(surfVal);
}

bool asFileGrib2::CheckProductDefinition(const gribfield *gfld) const
{
    if (gfld->ipdtnum != 0) {
        wxLogError(_("Only the Product Definition Template 4.0 is implemented so far."));
        return false;
    }

    // Product Definition Template 4.0 - http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_temp4-0.shtml
    if (gfld->ipdtmpl[7] != 1)
        return false;

    return true;
}

bool asFileGrib2::CheckGridDefinition(const gribfield *gfld) const
{
    if (gfld->igdtnum != 0) {
        wxLogError(_("Only the Grid Definition Template 3.0 is implemented so far."));
        return false;
    }

    // Grid Definition Template 3.0 - http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_temp3-0.shtml
    if (gfld->igdtmpl[0] != 6)
        return false;
    if (gfld->igdtmpl[1] != 0)
        return false;
    if (gfld->igdtmpl[2] != 0)
        return false;
    if (gfld->igdtmpl[3] != 0)
        return false;
    if (gfld->igdtmpl[4] != 0)
        return false;
    if (gfld->igdtmpl[5] != 0)
        return false;
    if (gfld->igdtmpl[6] != 0)
        return false;
    if (gfld->igdtmpl[9] != 0)
        return false;
    if (gfld->igdtmpl[10] != 0)
        return false;

    return true;
}

void asFileGrib2::BuildAxes(const gribfield *gfld)
{
    float scale = 0.000001;
    int nX = (int) gfld->igdtmpl[7];
    int nY = (int) gfld->igdtmpl[8];
    float latStart = float(gfld->igdtmpl[11]) * scale;
    float lonStart = float(gfld->igdtmpl[12]) * scale;
    float latEnd = float(gfld->igdtmpl[14]) * scale;
    float lonEnd = float(gfld->igdtmpl[15]) * scale;
    if (lonEnd < lonStart) {
        lonEnd += 360;
    }

    Array1DFloat xAxis = Array1DFloat::LinSpaced(nX, lonStart, lonEnd);
    Array1DFloat yAxis = Array1DFloat::LinSpaced(nY, latStart, latEnd);

    m_xAxes.push_back(xAxis);
    m_yAxes.push_back(yAxis);
}

void asFileGrib2::handleGribError(g2int ierr) const
{
    if (ierr == 1) {
        wxLogError(_("Beginning characters \"GRIB\" not found."));
    } else if (ierr == 2) {
        wxLogError(_("GRIB message is not Edition 2."));
    } else if (ierr == 3) {
        wxLogError(_("Could not find Section 1, where expected."));
    } else if (ierr == 4) {
        wxLogError(_("End string \"7777\" found, but not where expected."));
    } else if (ierr == 5) {
        wxLogError(_("End string \"7777\" not found at end of message."));
    } else if (ierr == 6) {
        wxLogError(_("Invalid section number found... OR"));
        wxLogError(_("... GRIB message did not contain the requested number of data fields."));
    } else if (ierr == 7) {
        wxLogError(_("End string \"7777\" not found at end of message."));
    } else if (ierr == 8) {
        wxLogError(_("Unrecognized Section encountered."));
    } else if (ierr == 9) {
        wxLogError(_("Data Representation Template 5.NN not yet implemented."));
    } else if (ierr >= 10 && ierr <= 16) {
        wxLogError(_("Error unpacking a Section."));
    } else {
        wxLogError(_("Unknown Grib error."));
    }
}

bool asFileGrib2::GetXaxis(Array1DFloat &uaxis) const
{
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);
    wxASSERT(m_xAxes.size() > m_index);

    uaxis = m_xAxes[m_index];

    return true;
}

bool asFileGrib2::GetYaxis(Array1DFloat &vaxis) const
{
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);
    wxASSERT(m_yAxes.size() > m_index);

    vaxis = m_yAxes[m_index];

    return true;
}

double asFileGrib2::GetTime() const
{
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);
    wxASSERT(m_times.size() > m_index);

    return m_times[m_index];
}

bool asFileGrib2::SetIndexPosition(const VectorInt gribCode, const float level)
{
    wxASSERT(gribCode.size() == 4);

    // Find corresponding data
    m_index = asNOT_FOUND;
    for (int i = 0; i < m_parameterNums.size(); ++i) {
        if (m_parameterDisciplines[i] == gribCode[0] && m_parameterCategories[i] == gribCode[1] &&
            m_parameterNums[i] == gribCode[2] && m_levelTypes[i] == gribCode[3], m_levels[i] == level) {

            if (m_index >= 0) {
                wxLogError(_("The desired parameter was found twice in the file."));
                return false;
            }

            m_index = i;
        }
    }

    if (m_index == asNOT_FOUND) {
        wxLogError(_("The desired parameter was not found in the file."));
        return false;
    }

    return true;
}

bool asFileGrib2::GetVarArray(const int IndexStart[], const int IndexCount[], float *pValue)
{
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);

    // Reposition stream position indicator
    if (fseek(m_filtPtr, m_messageOffsets[m_index], SEEK_SET) != 0) {
        wxLogError(_("Grib file read error."));
        return false;
    }

    // Read block of data from stream
    unsigned char *cgrib = (unsigned char *) malloc((size_t) m_messageSizes[m_index]);
    fread(cgrib, sizeof(unsigned char), m_messageSizes[m_index], m_filtPtr);

    // Get the data
    gribfield *gfld;
    int unpack = 1;
    g2int expand = 1;
    g2int ierr = g2_getfld(cgrib, m_fieldNum[m_index] + 1, unpack, expand, &gfld);
    if (ierr > 0) {
        handleGribError(ierr);
        return false;
    }

    if (gfld->unpacked != 1 || gfld->expanded != 1) {
        wxLogError(_("The Grib data were not unpacked neither expanded."));
        return false;
    }

    wxASSERT(gfld->ngrdpts > 0);
    wxASSERT(gfld->ngrdpts == m_xAxes[m_index].size() * m_yAxes[m_index].size());

    int iLonStart = IndexStart[0];
    int iLonEnd = IndexStart[0] + IndexCount[0] - 1;
    int iLatStart = IndexStart[1];
    int iLatEnd = IndexStart[1] + IndexCount[1] - 1;
    int nLons = (int) m_xAxes[m_index].size();
    int nLats = (int) m_yAxes[m_index].size();
    int finalIndex = 0;

    if (nLats > 0 && m_yAxes[m_index][0] > m_yAxes[m_index][1]) {
        for (int i_lat = nLats - 1; i_lat >= 0; i_lat--) {
            if (i_lat >= iLatStart && i_lat <= iLatEnd) {
                for (int i_lon = 0; i_lon < nLons; i_lon++) {
                    if (i_lon >= iLonStart && i_lon <= iLonEnd) {
                        pValue[finalIndex] = gfld->fld[i_lat * nLons + i_lon];
                        finalIndex++;
                    }
                }
            }
        }
    } else {
        for (int i_lat = 0; i_lat < nLats; i_lat++) {
            if (i_lat >= iLatStart && i_lat <= iLatEnd) {
                for (int i_lon = 0; i_lon < nLons; i_lon++) {
                    if (i_lon >= iLonStart && i_lon <= iLonEnd) {
                        pValue[finalIndex] = gfld->fld[i_lat * nLons + i_lon];
                        finalIndex++;
                    }
                }
            }
        }
    }

    g2_free(gfld);
    free(cgrib);

    return true;
}
