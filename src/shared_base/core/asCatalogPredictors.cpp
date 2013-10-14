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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asCatalogPredictors.h"

#include "wx/fileconf.h"

#include <asFileXml.h>


asCatalogPredictors::asCatalogPredictors(const wxString &alternateFilePath)
:
asCatalog(alternateFilePath)
{
    // Initiate some data
    m_Website = wxEmptyString;
    m_Ftp = wxEmptyString;
    m_Data.Id = wxEmptyString;
    m_Data.Name = wxEmptyString;
    m_Data.FileLength = Total;
    m_Data.FileName = wxEmptyString;
    m_Data.FileVarName = wxEmptyString;
}

asCatalogPredictors::~asCatalogPredictors()
{
    //dtor
}

bool asCatalogPredictors::Load(const wxString &DataSetId, const wxString &DataId)
{
    return false;
}

bool asCatalogPredictors::LoadDatasetProp(const wxString &DataSetId)
{
    return false;
}

bool asCatalogPredictors::LoadDataProp(const wxString &DataSetId, const wxString &DataId)
{
    return false;
}
