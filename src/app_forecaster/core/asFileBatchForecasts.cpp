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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#include "asFileBatchForecasts.h"

asFileBatchForecasts::asFileBatchForecasts(const wxString &FileName, const ListFileMode &FileMode)
:
asFileXml(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileBatchForecasts::~asFileBatchForecasts()
{
    //dtor
}

bool asFileBatchForecasts::EditRootElement()
{
    if (!GetRoot()) return false;
    GetRoot()->AddAttribute("target", "forecaster");
    return true;
}

bool asFileBatchForecasts::CheckRootElement()
{
    if (!GetRoot()) return false;
    if (!IsAnAtmoSwingFile()) return false;
    if (!FileVersionIsOrAbove(1.0)) return false;

    if (!GetRoot()->GetAttribute("target").IsSameAs("forecaster", false))
    {
        asLogError(wxString::Format(_("The file %s is not a parameters file for the Forecaster."), m_fileName.GetFullName()));
        return false;
    }
    return true;
}
