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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include <gtest/gtest.h>

#include "asFileText.h"

TEST(FileText, GetFileContent) {
    wxString filePath = wxFileName::GetCwd();
    filePath.Append("/files/precipitation_file.txt");
    asFileText file(filePath, asFile::ReadOnly);
    file.Open();

    wxString content = file.GetContent();

    EXPECT_TRUE(content.Len() > 0);
    EXPECT_TRUE(content.Contains("19620102  14.4"));
}
