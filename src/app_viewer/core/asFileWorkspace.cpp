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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asFileWorkspace.h"

asFileWorkspace::asFileWorkspace(const wxString& fileName, const FileMode& fileMode)
    : asFileXml(fileName, fileMode) {
    // FindAndOpen() processed by asFileXml
}

bool asFileWorkspace::EditRootElement() const {
    if (!GetRoot()) return false;
    GetRoot()->AddAttribute("target", "viewer");
    return true;
}

bool asFileWorkspace::CheckRootElement() const {
    if (!GetRoot()) return false;
    if (!IsAnAtmoSwingFile()) return false;
    if (!FileVersionIsOrAbove(1.0)) return false;

    if (!GetRoot()->GetAttribute("target").IsSameAs("viewer", false)) {
        wxLogError(_("The file %s is not a parameters file for the Viewer."), m_fileName.GetFullName());
        return false;
    }
    return true;
}
