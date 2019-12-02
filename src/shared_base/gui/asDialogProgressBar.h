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
 */

#ifndef AS_DIALOG_PROGRESS_BAR_H
#define AS_DIALOG_PROGRESS_BAR_H

#include "asIncludes.h"

#include "wx/progdlg.h"

class asDialogProgressBar : public wxObject {
 public:
  asDialogProgressBar(const wxString &dialogMessage, int valueMax);

  ~asDialogProgressBar() override;

  void Destroy();

  bool Update(int value, const wxString &message = wxEmptyString);

 protected:
 private:
  wxProgressDialog *m_progressBar;
  bool m_initiated;
  int m_steps;
  int m_delayUpdate;
  int m_valueMax;
  vi m_vectorSteps;
  int m_currentStepIndex;
};

#endif
