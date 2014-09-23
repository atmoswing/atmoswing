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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#ifndef ATMOSWINGMAINFORECATSER_H
#define ATMOSWINGMAINFORECATSER_H

//#include "version.h"
#include "asIncludes.h"
#include "AtmoswingAppCalibrator.h"
#include "asFrameCalibration.h"


class AtmoswingFrameCalibrator: public asFrameCalibration
{
public:
    AtmoswingFrameCalibrator(wxFrame *frame);
    ~AtmoswingFrameCalibrator();
private:
    virtual void OnClose(wxCloseEvent& event);
    virtual void OnQuit(wxCommandEvent& event);
    void ProcessTest();
    void SetDefaultOptions();
};

#endif // ATMOSWINGMAINFORECATSER_H