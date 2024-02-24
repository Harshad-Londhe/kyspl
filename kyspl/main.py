import threading
import multiprocessing
import time
import PySimpleGUI as sg
import neapr as ks


paths = ks.Paths(json_path='data.json',
dataset='model/keypoint_classifier/keypointtest.csv',
model_save_path ='model/keypoint_classifier/keypoint_classifiertest.hdf5',
tflite_save_path = 'model/keypoint_classifier/keypoint_classifiertest.tflite',
model_path='model/keypoint_classifier/keypoint_classifiertest.tflite')

class MyThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_thread = False

    def run(self):
        while not self.stop_thread:
            print("Thread running...")
            time.sleep(1)


    def stop(self):
        self.stop_thread = True


def getSpellData():
    spells=[]
    divider = '---------------------------------------------------------------------------------------'
    with ks.KeySpellList(paths) as ksl:

        for key_spell in ksl.get_key_spells():
            spells.append(divider)
            spells.append(f'{key_spell.name}:{key_spell.func}')
            #spells.append('----------------------')

    return spells, divider

spells, divider = getSpellData()

lb = sg.Listbox(values=spells, size=(20, 10), expand_x=True, no_scrollbar=True)
tb = sg.Text(''.join(spells))
# Define the layout of the window
layout = [
    [sg.Text('Select an item from the list:')],
    [lb],
    [sg.Button('Add Spell', key='ADDBUTTON'), sg.Button('Train Spell', key='TRAINBUTTON'), sg.Button('Start', key='STARTBUTTON'),
     sg.Button('Edit Spell', key='EDITBUTTON'), sg.Button('Delete Spell', key='DELETEBUTTON')],
    [sg.Text('Keyspells:'), sg.Text('Inactive', key='CAMSTATUS')]
]

# Create the window
window = sg.Window('KeySpells', layout)

def startcam():
    pass

def trainwindow(spelldata):
    spellname, spellfunc = spelldata[0].split(':')
    tb =sg.Text('', key='TRAINUPDATE')
    trainlay =[
        [sg.Text('You are about to train:')],
        [sg.Text(f'Spell: {spellname}')],
        [sg.Text(f'Func: {spellfunc}')],
        [tb],
        [sg.Button('Confirm', key='CONFIRMTRAIN'), sg.Button('Cancel', key='CANCELTRAIN')]
    ]
    trainwindow = sg.Window('Train spell', trainlay, modal=True)

    while True:
        event, values = trainwindow.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'CANCELTRAIN':
            break
        elif event == 'CONFIRMTRAIN':
            print(spellname,spellfunc)
            tb.update('Please while KeySpells is training.')
            with ks.KeySpellList(paths) as ksl:

                for key_spell in ksl.get_key_spells():
                    if key_spell.name == spellname:
                        print('training', spellname)

                        st = ks.SpellTrainer(ksl.get_key_spells(),key_spell, paths)
                        st.detect_hands()
                        print('training', spellname)
                        tb.update(f'Training completed for {spellname}')

def addwindow():
    addlay = [[sg.Text('Name:'), sg.Input(key='SPELLNAME')],
              [sg.Text('Function:'), sg.Input(key='SPELLFUNC')],
              [sg.Button('Brew', expand_x=True, key='ADDSPELL')]]
    addwindow = sg.Window("Add Spell", addlay)


    while True:
        event, values = addwindow.read()
        if event == sg.WIN_CLOSED:
            break

        elif event == 'ADDSPELL':
            pass
            spellname, spellfunc = values['SPELLNAME'], values['SPELLFUNC']
            with ks.KeySpellList(paths) as ksl:
                spell = ks.KeySpell(spellname, ksl.get_max_gid() + 1, spellfunc)
                ksl.add_key_spell(spell)
            spells, _ = getSpellData()
            lb.update(spells)


    addwindow.close()

def delwindow(spelldata):
    spell = spelldata[0]
    spellname, spellfunc = ''.join(spell).split(':')
    dellay = [[sg.Text("YOU ARE DELETING")],
              [sg.Text(spell,key='DELETESPELLNAME')],
              [sg.Button('OK', key='CONFIRMDELETE'), sg.Button('Cancel', key='CANCELDELETE')]]
    delwindow = sg.Window('Delete Spell', dellay, modal=True)
    while True:
        event, values = delwindow.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'CONFIRMDELETE':
            try:

                with ks.KeySpellList(paths) as ksl:
                    for key_spell in ksl.get_key_spells():
                        if key_spell.name == spellname:
                            ksl.remove_key_spell(key_spell)
            except ValueError:
                pass
            break
            #breakpoint()
        elif event == 'CANCELDELETE':
            break
    spells, _ = getSpellData()
    lb.update(spells)
    delwindow.close()

def editwindow(spelldata):
    spellname, spellfunc = spelldata[0].split(':')
    oldspellname, oldspellfunc = spelldata[0].split(':')
    editlay = [ [sg.Text(f'You are editing: {spellname}:{spellfunc}')],
              [sg.Text('Name:'), sg.Input(f'{spellname}',key='SPELLNAME'kk)],
              [sg.Text('Function:'), sg.Input(f'{spellfunc}',key='SPELLFUNC')],
              [sg.Button('Edit', expand_x=True, key='EDITSPELL')]]
    editwindow = sg.Window("Add Spell", editlay)


    while True:
        event, values = editwindow.read()
        if event == sg.WIN_CLOSED:
            break

        elif event == 'EDITSPELL':
            pass
            spellname, spellfunc = values['SPELLNAME'], values['SPELLFUNC']
            with ks.KeySpellList(paths) as ksl:
                for key_spell in ksl.get_key_spells():
                    if key_spell.name == oldspellname:
                        print(oldspellname, oldspellfunc)
                        print(spellname, spellfunc)
                        key_spell.name = spellname
                        key_spell.func = spellfunc
            spells, _ = getSpellData()
            lb.update(spells)


    editwindow.close()


# Event loop to process events and update the window
divider = '---------------------------------------------------------------------------------------'
start_app=True
camstatus=False


t=MyThread()
camth = threading.Thread(target=ks.Spellcaster(paths).wait_for_spell, args=True)
while start_app:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'ADDBUTTON':
        if lb.get() != divider:
            addwindow()

        #window['status'].update('Button 1 pressed')
    elif event == 'TRAINBUTTON':
        if lb.get() and lb.get()[0] != divider:
            spell = lb.get()
            trainwindow(spell)


        #window['status'].update('Button 2 pressed')
    elif event == 'STARTBUTTON':

        ks.Spellcaster(paths).wait_for_spell(True)
        #window['status'].update('Button 3 pressed')
    elif event == 'EDITBUTTON':
        if lb.get() and lb.get()[0] != divider:
            spell = lb.get()
            editwindow(spell)

        #window['status'].update('Button 4 pressed')
    elif event == 'DELETEBUTTON':
        if lb.get() and lb.get()[0] != divider:
            spell = lb.get()
            delwindow(spell)





# Close the window
window.close()

#star = ks.Spellcaster(paths)
#star.wait_for_spell(True)#