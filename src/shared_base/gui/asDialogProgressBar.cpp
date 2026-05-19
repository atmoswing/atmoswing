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

#include "asDialogProgressBar.h"

asDialogProgressBar::asDialogProgressBar(const wxString& dialogMessage, int valueMax)
    : _progressBar(nullptr),
      _initiated(false),
      _steps(100),
      _delayUpdate(false),
      _valueMax(valueMax),
      _currentStepIndex(0) {
    if (!g_silentMode) {
        if (valueMax > 2 * _steps) {
            _delayUpdate = true;
            _vectorSteps.resize(_steps + 1);
            for (int i = 0; i <= _steps; i++) {
                _vectorSteps[i] = i * valueMax / _steps;
            }
        }

        if (valueMax > 10) {
            _progressBar = new wxProgressDialog(_("Please wait"), dialogMessage, valueMax, nullptr,
                                                 wxPD_AUTO_HIDE | wxPD_CAN_ABORT | wxPD_REMAINING_TIME |
                                                     wxPD_ELAPSED_TIME | wxPD_SMOOTH);  // wxPD_APP_MODAL |
            _initiated = true;
        }
    }
}

asDialogProgressBar::~asDialogProgressBar() {
    if (_initiated) {
        _progressBar->Update(_valueMax);
        _progressBar->Destroy();
        _initiated = false;
        wxWakeUpIdle();
    }
}

void asDialogProgressBar::Destroy() {
    if (_initiated) {
        _progressBar->Update(_valueMax);
        _progressBar->Destroy();
        _initiated = false;
        wxWakeUpIdle();
    }
}

bool asDialogProgressBar::Update(int value, const wxString& message) {
    wxString newMessage = message;

    if (_initiated) {
        if (_delayUpdate) {
            if (value >= _vectorSteps[_currentStepIndex]) {
                _currentStepIndex++;
                if (g_verboseMode) {
                    if (!message.IsEmpty()) {
                        newMessage = message + asStrF("(%d/%d)", value, _valueMax);
                    }
                }
                return _progressBar->Update(value, newMessage);
            }
        } else {
            if (g_verboseMode) {
                if (!message.IsEmpty()) {
                    newMessage = message + asStrF("(%d/%d)", value, _valueMax);
                }
            }
            return _progressBar->Update(value, newMessage);
        }
    }
    return true;
}
