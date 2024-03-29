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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#ifndef AtmoSwingAPPVIEWER_H
#define AtmoSwingAPPVIEWER_H

#include <wx/app.h>
#include <wx/cmdline.h>
#include <wx/snglinst.h>
#include <wx/socket.h>

#include "asIncludes.h"

class asThreadsManager;

class AtmoSwingAppViewer : public wxApp {
  public:
    /**
     * The initialization of the application.
     */
    bool OnInit() override;

    /**
     * Clean up on exit.
     */
    int OnExit() override;

    /**
     * Initialize the command line parser.
     * 
     * @param parser The command line parser.
     * @note From http://wiki.wxwidgets.org/Command-Line_Arguments
     */
    void OnInitCmdLine(wxCmdLineParser& parser) override;

    /**
     * Proceed to the command line parsing.
     * 
     * @param parser The command line parser.
    */
    bool OnCmdLineParsed(wxCmdLineParser& parser) override;

    /**
     * Initialize the language support.
     */
    static void InitLanguageSupport();

  private:
    wxSingleInstanceChecker* m_singleInstanceChecker; /**< The single instance checker. */
};

DECLARE_APP(AtmoSwingAppViewer);

#endif
