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
 * Portions Copyright 2018-2019 Pascal Horton, University of Bern.
 */

#include "asFileGrib.h"

asFileGrib::asFileGrib(const wxString &fileName, const FileMode &fileMode)
        : asFile(fileName, fileMode),
          m_filtPtr(nullptr),
          m_handle(nullptr),
          m_index(asNOT_FOUND)
{
    switch (fileMode) {
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

asFileGrib::~asFileGrib()
{
    Close();
}

bool asFileGrib::Open()
{
    if (!Find())
        return false;

    // Let GDAL open the dataset
    if (!OpenDataset())
        return false;

    m_opened = true;

    return true;
}

bool asFileGrib::Close()
{
    if (m_filtPtr) {
        fclose(m_filtPtr);
        m_filtPtr = nullptr;
    }

    if (m_handle) {
        codes_handle_delete(m_handle);
        m_handle = nullptr;
    }

    return true;
}

bool asFileGrib::OpenDataset()
{
    // Filepath
    wxString filePath = m_fileName.GetFullPath();

    // Open file
    m_filtPtr = fopen(filePath.mb_str(), "r");

    if (!m_filtPtr) // Failed
    {
        wxLogError(_("The opening of the grib file failed."));
        wxFAIL;
        return false;
    }

    // Parse structure
    return ParseStructure();
}

bool asFileGrib::ParseStructure()
{





    return true;
}

bool asFileGrib::GetVarArray(const int IndexStart[], const int IndexCount[], float *pValue)
{
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);






    return true;
}
