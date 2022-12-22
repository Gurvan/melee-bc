package main

import (
	"compress/gzip"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/sbinet/npyio"
)

type ExternalCharacterID uint8

const (
	CaptainFalcon   ExternalCharacterID = 0x00
	DonkeyKong      ExternalCharacterID = 0x01
	Fox             ExternalCharacterID = 0x02
	GameAndWatch    ExternalCharacterID = 0x03
	Kirby           ExternalCharacterID = 0x04
	Bowser          ExternalCharacterID = 0x05
	Link            ExternalCharacterID = 0x06
	Luigi           ExternalCharacterID = 0x07
	Mario           ExternalCharacterID = 0x08
	Marth           ExternalCharacterID = 0x09
	Mewtwo          ExternalCharacterID = 0x0A
	Ness            ExternalCharacterID = 0x0B
	Peach           ExternalCharacterID = 0x0C
	Pikachu         ExternalCharacterID = 0x0D
	IceClimbers     ExternalCharacterID = 0x0E
	Jigglypuff      ExternalCharacterID = 0x0F
	Samus           ExternalCharacterID = 0x10
	Yoshi           ExternalCharacterID = 0x11
	Zelda           ExternalCharacterID = 0x12
	Sheik           ExternalCharacterID = 0x13
	Falco           ExternalCharacterID = 0x14
	YoungLink       ExternalCharacterID = 0x15
	DrMario         ExternalCharacterID = 0x16
	Roy             ExternalCharacterID = 0x17
	Pichu           ExternalCharacterID = 0x18
	Ganondorf       ExternalCharacterID = 0x19
	MasterHand      ExternalCharacterID = 0x1A
	WireframeMale   ExternalCharacterID = 0x1B
	WireframeFemale ExternalCharacterID = 0x1C
	GigaBowser      ExternalCharacterID = 0x1D
	CrazyHand       ExternalCharacterID = 0x1E
	Sandbag         ExternalCharacterID = 0x1F
	Popo            ExternalCharacterID = 0x20
	None            ExternalCharacterID = 0x21
)

type StageID uint8

const (
	Dummy             StageID = 0x00
	TEST              StageID = 0x01
	FountainOfDreams  StageID = 0x02
	PokemonStadium    StageID = 0x03
	PeachsCastle      StageID = 0x04
	KongoJungle       StageID = 0x05
	Brinstar          StageID = 0x06
	Corneria          StageID = 0x07
	YoshisStory       StageID = 0x08
	Onett             StageID = 0x09
	MuteCity          StageID = 0x0A
	RainbowCruise     StageID = 0x0B
	JungleJapes       StageID = 0x0C
	GreatBay          StageID = 0x0D
	HyruleTemple      StageID = 0x0E
	BrinstarDepths    StageID = 0x0F
	YoshisIsland      StageID = 0x10
	GreenGreens       StageID = 0x11
	Fourside          StageID = 0x12
	MushroomKingdomI  StageID = 0x13
	MushroomKingdomII StageID = 0x14
	Akaneia           StageID = 0x15
	Venom             StageID = 0x16
	PokeFloats        StageID = 0x17
	BigBlue           StageID = 0x18
	IcicleMountain    StageID = 0x19
	IceTop            StageID = 0x1A
	FlatZone          StageID = 0x1B
	DreamLandN64      StageID = 0x1C
	YoshisIslandN64   StageID = 0x1D
	KongoJungleN64    StageID = 0x1E
	Battlefield       StageID = 0x1F
	FinalDestination  StageID = 0x20
)

const (
	cmdPayloadSizes byte = 0x35
	cmdGameStart         = 0x36
	cmdPreFrame          = 0x37
	cmdPostFrame         = 0x38
)

// PreFrameUpdate is the output for a fighter as inputs are processed by the game
type PreFrameUpdate struct {
	JoystickX        float32
	JoystickY        float32
	CStickX          float32
	CStickY          float32
	Trigger          float32
	ProcessedButtons uint32
	RawX             uint8
}

// PostFrameUpdate is the output for a fighter after a frame is done processing before it gets displayed
type PostFrameUpdate struct {
	InternalCharID        uint8
	ActionStateID         uint16
	ActionStateCounter    float32
	PosX                  float32
	PosY                  float32
	SpeedX                float32
	SpeedY                float32
	FacingDirection       float32
	Percent               float32
	ShieldHealth          float32
	Grounded              bool
	JumpsRemaining        uint8
	HurtboxCollisionState uint8
	KnockbackSpeedX       float32
	KnockbackSpeedY       float32
	NextAnimationNumber   uint32
}

// PlayerFrame is a wrapper for a player's pre and post messages for a frame
type PlayerFrame struct {
	Pre  PreFrameUpdate
	Post PostFrameUpdate
}

// Frame contains all the information for a frame
type Frame struct {
	Number  int32
	Players [4]PlayerFrame
}

func (f Frame) Clone() Frame {
	var ff Frame
	ff.Number = f.Number
	for i, p := range f.Players {
		ff.Players[i] = p
	}
	return f
}

// Game is the top-level entry point for the contents of the slp file
type Game struct {
	SlippiVersion   [4]uint8
	ExternalCharIDs [4]uint8
	PlayerTypes     [4]uint8
	Costumes        [4]uint8
	UUIDs           [4]string
	StageIndex      uint16
	FrozenPS        bool
	Frames          []Frame
}

func (g Game) Clone() Game {
	var gg Game
	for i := 0; i < 4; i++ {
		gg.SlippiVersion[i] = g.SlippiVersion[i]
		gg.ExternalCharIDs[i] = g.ExternalCharIDs[i]
		gg.PlayerTypes[i] = g.PlayerTypes[i]
		gg.Costumes[i] = g.Costumes[i]
		gg.UUIDs[i] = g.UUIDs[i]
	}
	gg.StageIndex = g.StageIndex
	gg.FrozenPS = g.FrozenPS
	for _, f := range g.Frames {
		gg.Frames = append(gg.Frames, f.Clone())
	}
	return gg
}

func IsVersionGreaterOrEqual(version, other [4]uint8) bool {
	if version[0] < other[0] {
		return false
	} else if version[0] > other[0] {
		return true
	}
	if version[1] < other[1] {
		return false
	} else if version[1] > other[1] {
		return true
	}
	if version[2] < other[2] {
		return false
	} else if version[2] > other[2] {
		return true
	}
	if version[3] < other[3] {
		return false
	} else if version[3] > other[3] {
		return true
	}
	return true
}

func readUint32(buf []byte, pos int) uint32 {
	return binary.BigEndian.Uint32(buf[pos : pos+4])
}

func readUint16(buf []byte, pos int) uint16 {
	return binary.BigEndian.Uint16(buf[pos : pos+2])
}

func readUint8(buf []byte, pos int) uint8 {
	return uint8(buf[pos])
}

func readInt32(buf []byte, pos int) int32 {
	num := binary.BigEndian.Uint32(buf[pos : pos+4])
	return int32(num)
}

func readFloat(buf []byte, pos int) float32 {
	num := binary.BigEndian.Uint32(buf[pos : pos+4])
	return math.Float32frombits(num)
}

func readBool(buf []byte, pos int) bool {
	return buf[pos] > 0
}

func readString(buf []byte, pos int) string {
	n := 0
	for i, b := range buf[pos:] {
		if b == 0 {
			break
		}
		n = i
	}
	s := string(buf[pos : pos+n])
	return s
}

// Parse takes in a gzipped slp file path. It extracts and parses the slp file
func Parse(filePath string, headerOnly bool) (*Game, string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, "", err
	}
	defer file.Close()

	var res []byte
	nHeader := 1000
	if headerOnly {
		res = make([]byte, nHeader)
	}

	if strings.HasSuffix(filePath, ".gz") {
		gz, err := gzip.NewReader(file)
		if err != nil {
			return nil, "", err
		}
		defer gz.Close()
		if headerOnly {
			_, err = io.ReadAtLeast(gz, res, nHeader)
		} else {
			res, err = ioutil.ReadAll(gz)
		}
		if err != nil {
			return nil, "", err
		}
	} else {
		// res, err = ioutil.ReadAll(file)
		if headerOnly {
			_, err = io.ReadAtLeast(file, res, nHeader)
		} else {
			res, err = ioutil.ReadAll(file)
		}
		if err != nil {
			return nil, "", err
		}
	}

	checksum := sha1.Sum(res)
	checksumString := hex.EncodeToString(checksum[:])

	pos := 15

	// First load the payload sizes
	cmdByte := res[pos]
	if cmdByte != cmdPayloadSizes {
		return nil, "", errors.New("slp: first byte not payload sizes (0x35)")
	}
	pos++

	// Get payload size for payloadSizes
	payloadSizesSize := readUint8(res, pos)
	pos++

	// Load payload sizes
	payloadSizes := map[byte]uint16{}
	for pos < 16+int(payloadSizesSize) {
		payloadSizes[res[pos]] = readUint16(res, pos+0x1)
		pos += 3
	}

	// for cmdByte, size := range payloadSizes {
	//         fmt.Printf("[0x%x] %d\n", cmdByte, size)
	// }

	frames := []Frame{}
	framesByIndex := map[int32]*Frame{}

	getPlayer := func(res []byte, pos int) (*PlayerFrame, error) {
		frameNum := readInt32(res, pos+0x1)
		frame, ok := framesByIndex[frameNum]
		if !ok {
			frames = append(frames, Frame{})
			frame = &frames[len(frames)-1]
			framesByIndex[frameNum] = frame
		}

		frame.Number = frameNum

		playerIndex := readUint8(res, pos+0x5)
		if playerIndex > 3 {
			return &frame.Players[0], errors.New("Parsing error: Player Index")
		}
		player := &frame.Players[playerIndex]

		return player, nil
	}

	var version [4]uint8
	var stageIndex uint16
	var extCharIDs [4]uint8
	var playerTypes [4]uint8
	var costumes [4]uint8
	var uuids [4]string
	var frozenPS bool

	for pos < len(res) {
		cmdByte = res[pos]

		// Check if command byte is U which is the start of metadata
		if cmdByte == 0x55 {
			break
		}

		switch cmdByte {
		case cmdGameStart:
			for i := 0; i < 4; i++ {
				version[i] = readUint8(res, pos+0x1+i)
			}
			stageIndex = readUint16(res, pos+0x13)
			for i := 0; i < 4; i++ {
				extCharIDs[i] = readUint8(res, pos+0x5+0x60+i*0x24)
			}
			for i := 0; i < 4; i++ {
				playerTypes[i] = readUint8(res, pos+0x5+0x61+i*0x24)
			}
			for i := 0; i < 4; i++ {
				costumes[i] = readUint8(res, pos+0x5+0x63+i*0x24)
			}
			if IsVersionGreaterOrEqual(version, [4]uint8{3, 11, 0, 0}) {
				for i := 0; i < 4; i++ {
					uuids[i] = readString(res, pos+0x249+i*0x1D)
				}
			}

			frozenPS = readBool(res, pos+0x1A2)
			if headerOnly {
				break
			}
		case cmdPreFrame:
			if !headerOnly {
				player, err := getPlayer(res, pos)
				if err != nil {
					return nil, "", err
				}

				pre := PreFrameUpdate{}

				pre.JoystickX = readFloat(res, pos+0x19)
				pre.JoystickY = readFloat(res, pos+0x1D)
				pre.CStickX = readFloat(res, pos+0x21)
				pre.CStickY = readFloat(res, pos+0x25)
				pre.Trigger = readFloat(res, pos+0x29)
				pre.ProcessedButtons = readUint32(res, pos+0x2D)
				pre.RawX = readUint8(res, pos+0x3B)

				player.Pre = pre
			}
		case cmdPostFrame:
			if !headerOnly {
				player, err := getPlayer(res, pos)
				if err != nil {
					return nil, "", err
				}

				post := PostFrameUpdate{}

				post.InternalCharID = readUint8(res, pos+0x7)
				post.ActionStateID = readUint16(res, pos+0x8)
				post.ActionStateCounter = readFloat(res, pos+0x22)
				post.PosX = readFloat(res, pos+0xA)
				post.PosY = readFloat(res, pos+0xE)
				post.FacingDirection = readFloat(res, pos+0x12)
				post.Percent = readFloat(res, pos+0x16)
				post.ShieldHealth = readFloat(res, pos+0x1A)
				post.Grounded = !readBool(res, pos+0x2F)
				if IsVersionGreaterOrEqual(version, [4]uint8{2, 0, 0, 0}) {
					post.JumpsRemaining = readUint8(res, pos+0x32)
				} else {
					if post.Grounded {
						post.JumpsRemaining = 2
					} else {
						post.JumpsRemaining = 1
					}
				}
				if IsVersionGreaterOrEqual(version, [4]uint8{2, 1, 0, 0}) {
					post.HurtboxCollisionState = readUint8(res, pos+0x34)
				} else {
					post.HurtboxCollisionState = 0
				}
				if IsVersionGreaterOrEqual(version, [4]uint8{3, 5, 0, 0}) {
					airSpeed := readFloat(res, pos+0x35)
					groundSpeed := readFloat(res, pos+0x45)
					if !post.Grounded {
						post.SpeedX = airSpeed
					} else {
						post.SpeedX = groundSpeed
					}
					post.SpeedY = readFloat(res, pos+0x39)
					post.KnockbackSpeedX = readFloat(res, pos+0x3d)
					post.KnockbackSpeedY = readFloat(res, pos+0x41)
				} else {
					post.SpeedX = 0.
					post.SpeedX = 0.
					post.KnockbackSpeedX = 0.
					post.KnockbackSpeedY = 0.
				}
				if IsVersionGreaterOrEqual(version, [4]uint8{3, 10, 0, 0}) {
					post.NextAnimationNumber = readUint32(res, pos+0x4d)
				} else {
					post.NextAnimationNumber = 0
				}

				player.Post = post
			}
		}

		pos += int(payloadSizes[cmdByte]) + 1
	}

	return &Game{version, extCharIDs, playerTypes, costumes, uuids, stageIndex, frozenPS, frames}, checksumString, nil
}

func findFiles(root, ext string) []string {
	var a []string
	filepath.WalkDir(root, func(s string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if filepath.Ext(d.Name()) == ext {
			a = append(a, s)
		}
		return nil
	})
	return a
}

func findPortsByMainCharacter(g *Game, character ExternalCharacterID) []int {
	mainPort := -1
	altPort := -1
	for i, u := range g.ExternalCharIDs {
		if u == uint8(None) {
			continue
		} else if u == uint8(character) {
			mainPort = i
		} else {
			altPort = i
		}
	}

	// if mainPort == -1 || altPort == -1 || mainPort == altPort {
	//         return nil
	// }
	ports := []int{mainPort, altPort}
	return ports
}

func findPortsByUUID(g *Game, uuid string) []int {
	mainPort := -1
	altPort := -1
	for i, u := range g.UUIDs {
		if u == "" {
			continue
		} else if u == uuid {
			mainPort = i
		} else {
			altPort = i
		}
	}

	// if mainPort == -1 || altPort == -1 || mainPort == altPort {
	//         return nil
	// }
	ports := []int{mainPort, altPort}
	return ports
}

func filterByMainCharacter(g *Game, character ExternalCharacterID) bool {
	if character == None {
		return true
	}
	ports := findPortsByMainCharacter(g, character)
	if ports[0] == -1 {
		return false
	}
	return true
}

func filterByUUID(g *Game, uuid string) bool {
	if uuid == "" {
		return true
	}
	ports := findPortsByUUID(g, uuid)
	if ports[0] == -1 {
		return false
	}
	return true
}

func parseSlp(path string, allowedStages map[StageID]bool, allowedCharacters map[ExternalCharacterID]bool, mainUUID string) (*Game, string) {
	var g *Game = nil
	g, checksum, err := Parse(path, true)
	if err != nil {
		log.Println(err)
		return nil, ""
	}

	if !filterByUUID(g, mainUUID) {
		// fmt.Println("UUID not found")
		return nil, ""
	}

	// Check stage
	if allow, ok := allowedStages[StageID(g.StageIndex)]; !ok {
		// fmt.Println(path)
		// fmt.Println("Unallowed stage")
		return nil, ""
	} else {
		if !allow {
			return nil, ""
		}
	}

	// Check PS frozen
	if g.StageIndex == uint16(PokemonStadium) {
		if !g.FrozenPS {
			return nil, ""
		}
	}

	// Check players
	ports := make(map[int]bool)
	for i, pt := range g.PlayerTypes {
		if pt == 0 {
			ports[i] = true
		}
	}
	if len(ports) != 2 {
		return nil, ""
	}

	// Check characters
	for k := range ports {
		character := g.ExternalCharIDs[k]
		if allow, ok := allowedCharacters[ExternalCharacterID(character)]; !ok {
			// fmt.Println("Unallowed character")
			return nil, ""
		} else {
			if !allow {
				return nil, ""
			}
		}
	}
	g, checksum, err = Parse(path, false)
	if err != nil {
		log.Println(err)
		return nil, ""
	}
	return g, checksum
}

// Sticks
// var stickValues [117]float32 = [117]float32{
//         -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68,
//         -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55,
//         -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42,
//         -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29,
//         -28, -27, -26, -25, -24, -23, 0, 23, 24, 25, 26, 27, 28,
//         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
//         42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
//         55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
//         68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
// }

var stickBins [116]float32 = [116]float32{
	-79.5, -78.5, -77.5, -76.5, -75.5, -74.5, -73.5, -72.5, -71.5,
	-70.5, -69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5, -62.5,
	-61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5, -53.5,
	-52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5, -45.5, -44.5,
	-43.5, -42.5, -41.5, -40.5, -39.5, -38.5, -37.5, -36.5, -35.5,
	-34.5, -33.5, -32.5, -31.5, -30.5, -29.5, -28.5, -27.5, -26.5,
	-25.5, -24.5, -23.5, -22.5, 0.5, 23.5, 24.5, 25.5, 26.5,
	27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5,
	36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5,
	45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5,
	54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5,
	63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5,
	72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5,
}

// var triggerValues [99]float32 = [99]float32{
//         0, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
//         55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
//         68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
//         81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
//         94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
//         107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
//         120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
//         133, 134, 135, 136, 137, 138, 139, 140,
// }

var triggerBins [98]float32 = [98]float32{
	0.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5,
	51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5,
	60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5,
	69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5,
	78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5,
	87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5,
	96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5,
	105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5,
	114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5, 122.5,
	123.5, 124.5, 125.5, 126.5, 127.5, 128.5, 129.5, 130.5, 131.5,
	132.5, 133.5, 134.5, 135.5, 136.5, 137.5, 138.5, 139.5,
}

// var ascounterValues [121]float32 = [121]float32{
//         -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
//         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
//         25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
//         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
//         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
//         64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
//         77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
//         90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
//         103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
//         116, 117, 118, 119,
// }

var ascounterBins [121]float32 = [121]float32{
	-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
	8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5,
	17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5,
	26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5,
	35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5,
	44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5,
	53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5,
	62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5,
	71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5,
	80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5,
	89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5,
	98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5,
	107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5,
	116.5, 117.5, 118.5, 119.5,
}

// var damageValues [150]float32 = [150]float32{
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
//         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
//         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
//         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
//         52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
//         65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
//         78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
//         91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
//         104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
//         117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
//         130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
//         143, 144, 145, 146, 147, 148, 149,
// }

var damageBins [150]float32 = [150]float32{
	0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
	9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
	18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5,
	27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5,
	36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5,
	45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5,
	54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5,
	63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5,
	72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5,
	81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5,
	90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5,
	99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5,
	108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5,
	117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5,
	126.5, 127.5, 128.5, 129.5, 130.5, 131.5, 132.5, 133.5, 134.5,
	135.5, 136.5, 137.5, 138.5, 139.5, 140.5, 141.5, 142.5, 143.5,
	144.5, 145.5, 146.5, 147.5, 148.5, 149.5,
}

var posXBins [300]float32 = [300]float32{
	-149.5, -148.5, -147.5, -146.5, -145.5, -144.5, -143.5, -142.5,
	-141.5, -140.5, -139.5, -138.5, -137.5, -136.5, -135.5, -134.5,
	-133.5, -132.5, -131.5, -130.5, -129.5, -128.5, -127.5, -126.5,
	-125.5, -124.5, -123.5, -122.5, -121.5, -120.5, -119.5, -118.5,
	-117.5, -116.5, -115.5, -114.5, -113.5, -112.5, -111.5, -110.5,
	-109.5, -108.5, -107.5, -106.5, -105.5, -104.5, -103.5, -102.5,
	-101.5, -100.5, -99.5, -98.5, -97.5, -96.5, -95.5, -94.5,
	-93.5, -92.5, -91.5, -90.5, -89.5, -88.5, -87.5, -86.5,
	-85.5, -84.5, -83.5, -82.5, -81.5, -80.5, -79.5, -78.5,
	-77.5, -76.5, -75.5, -74.5, -73.5, -72.5, -71.5, -70.5,
	-69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5, -62.5,
	-61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5,
	-53.5, -52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5,
	-45.5, -44.5, -43.5, -42.5, -41.5, -40.5, -39.5, -38.5,
	-37.5, -36.5, -35.5, -34.5, -33.5, -32.5, -31.5, -30.5,
	-29.5, -28.5, -27.5, -26.5, -25.5, -24.5, -23.5, -22.5,
	-21.5, -20.5, -19.5, -18.5, -17.5, -16.5, -15.5, -14.5,
	-13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5,
	-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5,
	2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
	10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
	18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5,
	26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5,
	34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5,
	42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5,
	50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5,
	58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5,
	66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5,
	74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5,
	82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5,
	90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5,
	98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5,
	106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5,
	114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5,
	122.5, 123.5, 124.5, 125.5, 126.5, 127.5, 128.5, 129.5,
	130.5, 131.5, 132.5, 133.5, 134.5, 135.5, 136.5, 137.5,
	138.5, 139.5, 140.5, 141.5, 142.5, 143.5, 144.5, 145.5,
	146.5, 147.5, 148.5, 149.5,
}

var posYBins [125]float32 = [125]float32{
	-49.5, -48.5, -47.5, -46.5, -45.5, -44.5, -43.5, -42.5, -41.5,
	-40.5, -39.5, -38.5, -37.5, -36.5, -35.5, -34.5, -33.5, -32.5,
	-31.5, -30.5, -29.5, -28.5, -27.5, -26.5, -25.5, -24.5, -23.5,
	-22.5, -21.5, -20.5, -19.5, -18.5, -17.5, -16.5, -15.5, -14.5,
	-13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5,
	-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5,
	4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
	13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5,
	22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5,
	31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5,
	40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5,
	49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5,
	58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5,
	67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5,
}

var shieldHealthBins [6]float32 = [6]float32{
	0.5, 10.5, 20.5, 30.5, 40.5, 50.5,
}

type ButtonBits uint32

const (
	buttonDLeft  ButtonBits = 1
	buttonDRight            = 1 << 1
	buttonDDown             = 1 << 2
	buttonDUp               = 1 << 3
	buttonZ                 = 1 << 4
	buttonR                 = 1 << 5
	buttonL                 = 1 << 6
	buttonA                 = 1 << 8
	buttonB                 = 1 << 9
	buttonX                 = 1 << 10
	buttonY                 = 1 << 11
	buttonStart             = 1 << 12
)

func digitize(x float32, bins_ []float32) uint16 {
	bins := make([]float64, len(bins_))
	for i, b := range bins_ {
		bins[i] = float64(b)
	}
	return uint16(sort.SearchFloat64s(bins, float64(x)))
}

func digitizeController(p *PreFrameUpdate) []uint16 {
	data := make([]uint16, 0)
	data = append(data, digitize(80*p.JoystickX, stickBins[:]))
	data = append(data, digitize(80*p.JoystickY, stickBins[:]))
	data = append(data, digitize(80*p.CStickX, stickBins[:]))
	data = append(data, digitize(80*p.CStickY, stickBins[:]))
	data = append(data, digitize(140*p.Trigger, triggerBins[:]))

	// Z, L/R, A, B, X/Y
	var Z uint16 = 0
	if p.ProcessedButtons&buttonZ != 0 {
		Z = 1
	}
	data = append(data, Z)

	var LR uint16 = 0
	if p.ProcessedButtons&(buttonL|buttonR) != 0 {
		LR = 1
	}
	data = append(data, LR)

	var A uint16 = 0
	if p.ProcessedButtons&buttonA != 0 {
		A = 1
	}
	data = append(data, A)

	var B uint16 = 0
	if p.ProcessedButtons&buttonB != 0 {
		B = 1
	}
	data = append(data, B)

	var XY uint16 = 0
	if p.ProcessedButtons&(buttonX|buttonY) != 0 {
		XY = 1
	}
	data = append(data, XY)
	return data
}

// var maxASID uint16 = 0

func digitizeState(p *PostFrameUpdate) []uint16 {
	data := make([]uint16, 0)
	data = append(data, p.ActionStateID)
	// if p.ActionStateID > maxASID {
	//         maxASID = p.ActionStateID
	//         fmt.Println(p.ActionStateID)
	// }
	data = append(data, digitize(p.ActionStateCounter, ascounterBins[:]))
	data = append(data, digitize(p.Percent, damageBins[:]))
	data = append(data, uint16((p.FacingDirection+1)/2))
	data = append(data, digitize(p.PosX, posXBins[:]))
	data = append(data, digitize(p.PosY, posYBins[:]))
	data = append(data, digitize(p.ShieldHealth, shieldHealthBins[:]))
	if p.Grounded {
		data = append(data, 1)
	} else {
		data = append(data, 0)
	}
	data = append(data, uint16(p.JumpsRemaining))
	data = append(data, uint16(p.HurtboxCollisionState))
	return data
}

func extractController(p *PreFrameUpdate) []float32 {
	// returns 12 float32
	data := make([]float32, 0, 12)
	data = append(data, 80*p.JoystickX)
	data = append(data, 80*p.JoystickY)
	data = append(data, 80*p.CStickX)
	data = append(data, 80*p.CStickY)
	data = append(data, 140*p.Trigger)

	// Z, L, R, A, B, X, Y
	var Z float32 = 0
	if p.ProcessedButtons&buttonZ != 0 {
		Z = 1
	}
	data = append(data, Z)

	var L float32 = 0
	if p.ProcessedButtons&buttonL != 0 {
		L = 1
	}
	data = append(data, L)

	var R float32 = 0
	if p.ProcessedButtons&buttonR != 0 {
		R = 1
	}
	data = append(data, R)

	var A float32 = 0
	if p.ProcessedButtons&buttonA != 0 {
		A = 1
	}
	data = append(data, A)

	var B float32 = 0
	if p.ProcessedButtons&buttonB != 0 {
		B = 1
	}
	data = append(data, B)

	var X float32 = 0
	if p.ProcessedButtons&buttonX != 0 {
		X = 1
	}
	data = append(data, X)

	var Y float32 = 0
	if p.ProcessedButtons&buttonY != 0 {
		Y = 1
	}
	data = append(data, Y)
	return data
}

func extractState(p *PostFrameUpdate) []float32 {
	// returns 9 float32
	data := make([]float32, 0)
	data = append(data, float32(p.InternalCharID))
	data = append(data, float32(p.ActionStateID))
	// if p.ActionStateID > maxASID {
	//         maxASID = p.ActionStateID
	//         fmt.Println(p.ActionStateID)
	// }
	data = append(data, p.ActionStateCounter)
	data = append(data, p.Percent)
	data = append(data, p.FacingDirection)
	data = append(data, p.PosX)
	data = append(data, p.PosY)
	data = append(data, p.ShieldHealth)
	// if p.Grounded {
	//         data = append(data, 1)
	// } else {
	//         data = append(data, 0)
	// }
	data = append(data, float32(p.JumpsRemaining))
	// data = append(data, float32(p.HurtboxCollisionState))
	return data
}

func convertGameToSlice(g *Game, mainUUID string) []float32 {
	headerSize := 3
	numPlayers := 2
	controllerSize := 12
	stateSize := 9
	n := numPlayers*(controllerSize+stateSize)*len(g.Frames) + headerSize
	data := make([]float32, 0, n)

	var ports []int
	if mainUUID == "" {
		ports = make([]int, 0, 2)
		for i, pt := range g.PlayerTypes {
			if pt == 0 {
				ports = append(ports, i)
			}
		}
	} else {
		ports = findPortsByUUID(g, mainUUID)
	}

	data = append(data, float32(g.StageIndex))
	for _, port := range ports {
		data = append(data, float32(g.ExternalCharIDs[port]))
	}

	for _, f := range g.Frames {
		// fmt.Printf("%04d | 0x%03X | %5.1f\n", f.Number, f.Players[0].Post.ActionStateID, f.Players[0].Post.ActionStateCounter)
		// if f.Number > 200 {
		//         break
		// }
		for _, port := range ports {
			// fmt.Println(port)
			// fmt.Printf("%04d | %d | 0x%03X | %5.1f\n", f.Number, port, f.Players[port].Post.ActionStateID, f.Players[port].Post.ActionStateCounter)
			p := f.Players[port]
			// Controller

			k := len(data)
			data = append(data, extractController(&p.Pre)...)

			// State
			data = append(data, extractState(&p.Post)...)
			if len(data) != k+controllerSize+stateSize {
				log.Fatal()
			}
		}
	}
	if len(data) != n {
		log.Fatal()
	}
	return data
}

func convertGameToSliceDigitized(g *Game, mainUUID string) []uint16 {
	n := 2*(10+10)*len(g.Frames) + 3
	data := make([]uint16, 0, n)

	var ports []int
	if mainUUID == "" {
		ports = make([]int, 0, 2)
		for i, pt := range g.PlayerTypes {
			if pt == 0 {
				ports = append(ports, i)
			}
		}
	} else {
		ports = findPortsByUUID(g, mainUUID)
	}

	data = append(data, uint16(g.StageIndex))
	for _, port := range ports {
		data = append(data, uint16(g.ExternalCharIDs[port]))
	}

	for _, f := range g.Frames {
		for _, port := range ports {
			// fmt.Println(port)
			p := f.Players[port]
			// Controller
			data = append(data, digitizeController(&p.Pre)...)

			// State
			data = append(data, digitizeState(&p.Post)...)
		}
	}
	return data
}

func writeNpy[N uint16 | float32](path string, data []N) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	err = npyio.Write(f, data)
	if err != nil {
		return err
	}

	err = f.Close()
	if err != nil {
		return err
	}
	return nil
}

func main() {
	var slpDir, outputDir string
	var N int

	flag.StringVar(&slpDir, "slp_dir", "", "directory containing SLP files")
	flag.StringVar(&slpDir, "i", "", "directory containing SLP files")
	flag.StringVar(&outputDir, "output_dir", "", "directory to write output to")
	flag.StringVar(&outputDir, "o", "", "directory to write output to")
	flag.IntVar(&N, "N", 1, "number of CPU threads to use")

	// Parse the command-line flags.
	flag.Parse()

	// Use MkdirAll to create the output directory, including any missing parent directories.
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		fmt.Println(err)
		return
	}

	allowedCharacters := map[ExternalCharacterID]bool{
		CaptainFalcon: true,
		DonkeyKong:    true,
		Fox:           true,
		GameAndWatch:  true,
		Kirby:         true,
		Bowser:        true,
		Link:          true,
		Luigi:         true,
		Mario:         true,
		Marth:         true,
		Mewtwo:        true,
		Ness:          true,
		Peach:         true,
		Pikachu:       true,
		Jigglypuff:    true,
		Samus:         true,
		Yoshi:         true,
		Falco:         true,
		YoungLink:     true,
		DrMario:       true,
		Roy:           true,
		Pichu:         true,
		Ganondorf:     true,
		Zelda:         true,
		Sheik:         true,
		IceClimbers:   true,
	}

	allowedStages := map[StageID]bool{
		FountainOfDreams: true,
		PokemonStadium:   true,
		YoshisStory:      true,
		DreamLandN64:     true,
		Battlefield:      true,
		FinalDestination: true,
	}

	mainUUID := ""

	paths := findFiles(slpDir, ".slp")

	pathChan := make(chan string)
	retChan := make(chan int)
	var wg sync.WaitGroup
	convertFileInner := func(path string) int {
		defer func() {
			if r := recover(); r != nil {
				fmt.Println("Recovered in convertFile", r)
				retChan <- 0
			}
		}()
		// fmt.Println(path)
		g, checksum := parseSlp(path, allowedStages, allowedCharacters, mainUUID)
		_ = checksum
		if g == nil {
			return 0
		}
		// fmt.Println(g.SlippiVersion)
		// data := convertGameToSliceDigitized(g, mainUUID)
		data := convertGameToSlice(g, mainUUID)
		filename := checksum + ".npy"
		npyPath := filepath.Join(outputDir, filename)
		err := writeNpy(npyPath, data)
		if err != nil {
			log.Println(err)
		}
		return 1
	}
	convertFile := func() {
		defer wg.Done()

		for {
			path := <-pathChan
			if path == "stop" {
				return
			}
			retChan <- convertFileInner(path)
		}
	}
	// go convertFile()

	// N := 16
	fmt.Println("Num threads %d", N)
	go func() {
		for _, path := range paths {
			pathChan <- path
		}
		for i := 0; i < N; i++ {
			pathChan <- "stop"
		}
	}()
	for i := 0; i < N; i++ {
		wg.Add(1)
		go convertFile()
	}

	c := 0
	for n := 0; n < len(paths); n++ {
		c += <-retChan
		fmt.Printf("%d/%d/%d/%.2f%%\n", c, n+1, len(paths), 100.*float32(n+1)/float32(len(paths)))
	}

	wg.Wait()
}
